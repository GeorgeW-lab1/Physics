// Raspberry Pi Pico / RP2040 - Kaspa kHeavyHash optical bench, standalone.
// Build with the Earle Philhower Arduino-Pico core.
//
// Everything that used to live in kaspa_bench_offload_pico.py now runs on the Pico:
//   - job setup (demo template / pre-PoW hash / 80-byte work order, target by
//     leading zero bits or exact hex)
//   - matrix generation + GF(2) full-rank check
//   - nonce search (cSHAKE256 -> 64x64 matrix -> XOR -> cSHAKE256 -> target compare)
//   - independent re-verification of any hit (second Keccak/cSHAKE/matrix
//     implementation with runtime-derived constants)
//   - ITO PWM updates, photodiode / camera sampling, CSV rows, JSON report
//
// With AUTO_RUN_ON_BOOT=1 it runs the demo job by itself at power-up and prints
// the report over USB serial. A PC is optional (serial monitor, 115200 baud).
//
// Wiring:
//   Photodiodes: 8:1 74HC4051 mux COM -> GP26/ADC0, select S0/S1/S2 -> GP10/11/12
//   ITO pixels : GP2..GP9 (PWM; use external drivers)
//
// Serial commands (all optional):
//   PING
//   MODE PHOTODIODES | MODE CAMERA | MODE NONE
//   PREPOW <64 hex>            WORKORDER <160 hex>      DEMO
//   TIMESTAMP <hex or decimal> TARGETBITS <1..255>      TARGET <64 hex>
//   START <n>  MAXNONCES <n>   BATCH <n>   ITOEVERY <n>
//   RUN                         run a full search with the current settings
//   STOP is any byte sent while RUN is active
//   JOB <pre_pow 64h> <ts 16h> <target 64h> <start 16h> <count 16h>   (legacy, matrix from MATRIXROW)
//   MATRIXROW <0..63> <128 hex>   GENMATRIX   SAMPLE   SETITO a b c d e f g h   ITOCLEAR
//
// Replies: OK,... ERR,... D,... F,... CSV,... FOUND,... NONE,... REPORT,{json}
//
// Optical readings are telemetry only. They do not perform or validate the hash.

#include <Arduino.h>
#include <string.h>
#include <strings.h>

#define AUTO_RUN_ON_BOOT 1
#define RATE 136
#define NUM_ITO 8
#define NUM_PD 8

static const int PHOTO_ADC_PIN = 26;
static const int MUX_SELECT_PINS[3] = {10, 11, 12};
static const int ITO_PINS[NUM_ITO] = {2, 3, 4, 5, 6, 7, 8, 9};

// ---------------------------------------------------------------------------
// Settings / state
// ---------------------------------------------------------------------------

enum Source { SRC_NONE, SRC_PHOTODIODES, SRC_CAMERA };

static Source source = SRC_PHOTODIODES;
static uint8_t matrix[64][64];
static bool matrix_loaded = false;

static uint8_t pre_pow[32];
static uint8_t target[32];
static uint64_t timestamp = 1700000000ULL;  // no RTC on Pico; set with TIMESTAMP
static uint64_t start_nonce = 0;
static uint64_t max_nonces = 1000000ULL;
static uint32_t batch_size = 1024;           // progress/STOP polling interval
static uint32_t ito_every = 1;
static bool matrix_attempt_info = false;

static char line_buf[400];
static size_t line_len = 0;

// ---------------------------------------------------------------------------
// Optics
// ---------------------------------------------------------------------------

static uint16_t sample_pd(int channel) {
  channel &= 7;
  for (int bit = 0; bit < 3; ++bit) digitalWrite(MUX_SELECT_PINS[bit], (channel >> bit) & 1);
  delayMicroseconds(8);
  return analogRead(PHOTO_ADC_PIN);
}

static void update_optics(const uint8_t h[32]) {
  for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], h[i]);
}

static void clear_optics() {
  for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], 0);
}

// ---------------------------------------------------------------------------
// Keccak-f[1600] (fast table version)
// ---------------------------------------------------------------------------

static const uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
  0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
  0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
  0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
  0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
  0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
  0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int ROTC[24] = {
  1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8,
  25, 43, 62, 18, 39, 61, 20, 44
};

static const int PILN[24] = {
  10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13,
  12, 2, 20, 14, 22, 9, 6, 1
};

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static void keccakf(uint64_t st[25]) {
  uint64_t t, bc[5];
  for (int round = 0; round < 24; round++) {
    for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    t = st[1];
    for (int i = 0; i < 24; i++) {
      int j = PILN[i];
      bc[0] = st[j];
      st[j] = ROL64(t, ROTC[i]);
      t = bc[0];
    }
    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; i++) bc[i] = st[j + i];
      for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    st[0] ^= RC[round];
  }
}

// ---------------------------------------------------------------------------
// cSHAKE256 (N = "", S = custom)
// ---------------------------------------------------------------------------

static void store64_le(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (v >> (8 * i)) & 0xff;
}

static void xor_block(uint64_t st[25], const uint8_t *block, size_t len) {
  uint8_t *sb = (uint8_t *)st;
  for (size_t i = 0; i < len; i++) sb[i] ^= block[i];
}

// Absorb the final message piece with cSHAKE domain padding (0x04) and permute.
static void keccak_absorb_final(uint64_t st[25], const uint8_t *data, size_t len) {
  while (len >= RATE) {
    xor_block(st, data, RATE);
    keccakf(st);
    data += RATE;
    len -= RATE;
  }
  uint8_t block[RATE];
  memset(block, 0, RATE);
  memcpy(block, data, len);
  block[len] ^= 0x04;
  block[RATE - 1] ^= 0x80;
  xor_block(st, block, RATE);
  keccakf(st);
}

static void keccak_squeeze(uint64_t st[25], uint8_t *out, size_t outlen) {
  size_t made = 0;
  while (made < outlen) {
    size_t chunk = outlen - made;
    if (chunk > RATE) chunk = RATE;
    memcpy(out + made, (uint8_t *)st, chunk);
    made += chunk;
    if (made < outlen) keccakf(st);
  }
}

static size_t left_encode(uint8_t *out, size_t x) {
  uint8_t tmp[8];
  size_t n = 0;
  while (x > 0) { tmp[n++] = x & 0xff; x >>= 8; }
  if (n == 0) { out[0] = 1; out[1] = 0; return 2; }
  out[0] = n;
  for (size_t i = 0; i < n; i++) out[1 + i] = tmp[n - 1 - i];
  return n + 1;
}

// State after absorbing bytepad(encode_string("") || encode_string(custom), RATE).
// This is one full block, absorbed WITHOUT extra padding (per NIST SP 800-185).
static void cshake_init(uint64_t st[25], const char *custom) {
  memset(st, 0, 25 * sizeof(uint64_t));
  uint8_t prefix[RATE];
  memset(prefix, 0, RATE);
  size_t p = 0;
  p += left_encode(prefix + p, RATE);
  p += left_encode(prefix + p, 0);
  size_t clen = strlen(custom);
  p += left_encode(prefix + p, clen * 8);
  memcpy(prefix + p, custom, clen);
  xor_block(st, prefix, RATE);
  keccakf(st);
}

static void cshake256(const uint8_t *data, size_t datalen, const char *custom, uint8_t *out, size_t outlen) {
  uint64_t st[25];
  cshake_init(st, custom);
  keccak_absorb_final(st, data, datalen);
  keccak_squeeze(st, out, outlen);
}

// Precomputed post-prefix states: saves one Keccak permutation per cSHAKE call.
static uint64_t st_pow0[25];
static uint64_t st_heavy0[25];

static void cshake_fast(const uint64_t st0[25], const uint8_t *data, size_t datalen, uint8_t out[32]) {
  uint64_t st[25];
  memcpy(st, st0, sizeof(st));
  keccak_absorb_final(st, data, datalen);  // datalen <= 48 here: one block
  memcpy(out, (uint8_t *)st, 32);
}

// ---------------------------------------------------------------------------
// kHeavyHash
// ---------------------------------------------------------------------------

static void heavy_transform(const uint8_t pow_hash[32], uint8_t out[32]) {
  uint16_t vector[64];
  for (int i = 0; i < 32; i++) {
    vector[2 * i] = pow_hash[i] >> 4;
    vector[2 * i + 1] = pow_hash[i] & 0x0f;
  }
  for (int i = 0; i < 64; i += 2) {
    uint32_t a = 0, b = 0;
    for (int j = 0; j < 64; j++) {
      a += (uint32_t)matrix[i][j] * vector[j];
      b += (uint32_t)matrix[i + 1][j] * vector[j];
    }
    uint8_t prod = (uint8_t)((((a >> 10) & 0x0f) << 4) | ((b >> 10) & 0x0f));
    out[i / 2] = pow_hash[i / 2] ^ prod;
  }
}

static void kheavy_hash(uint64_t nonce, uint8_t final_hash[32]) {
  uint8_t work[48], pow_hash[32], transformed[32];
  memcpy(work, pre_pow, 32);
  store64_le(work + 32, timestamp);
  store64_le(work + 40, nonce);
  cshake_fast(st_pow0, work, 48, pow_hash);
  heavy_transform(pow_hash, transformed);
  cshake_fast(st_heavy0, transformed, 32, final_hash);
  for (int i = 0; i < 16; i++) {  // Kaspa compares the byte-reversed hash
    uint8_t t = final_hash[i];
    final_hash[i] = final_hash[31 - i];
    final_hash[31 - i] = t;
  }
}

static bool hash_meets_target(const uint8_t h[32]) {
  for (int i = 0; i < 32; i++) {
    if (h[i] < target[i]) return true;
    if (h[i] > target[i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Matrix generation (same scheme as the Python bench) + GF(2) rank
// ---------------------------------------------------------------------------

static int gf2_rank_64() {
  uint64_t rows[64];
  for (int i = 0; i < 64; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 64; j++) if (matrix[i][j] & 1) v |= (1ULL << j);
    rows[i] = v;
  }
  int rank = 0;
  for (int col = 0; col < 64; col++) {
    int pivot = -1;
    for (int r = rank; r < 64; r++) if ((rows[r] >> col) & 1) { pivot = r; break; }
    if (pivot < 0) continue;
    uint64_t tmp = rows[rank]; rows[rank] = rows[pivot]; rows[pivot] = tmp;
    for (int r = 0; r < 64; r++) if (r != rank && ((rows[r] >> col) & 1)) rows[r] ^= rows[rank];
    rank++;
  }
  return rank;
}

// cSHAKE256(seed || attempt_le32, "KaspaBenchMatrix"), 4096-byte stream, low nibbles.
static bool generate_matrix(int max_attempts, int *attempts_used) {
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    uint8_t seed[36];
    memcpy(seed, pre_pow, 32);
    seed[32] = attempt & 0xff;
    seed[33] = (attempt >> 8) & 0xff;
    seed[34] = (attempt >> 16) & 0xff;
    seed[35] = (attempt >> 24) & 0xff;

    uint64_t st[25];
    cshake_init(st, "KaspaBenchMatrix");
    keccak_absorb_final(st, seed, 36);

    int pos = 0;
    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < 64; j++) {
        if (pos == RATE) { keccakf(st); pos = 0; }
        matrix[i][j] = ((uint8_t *)st)[pos++] & 0x0f;
      }
    }
    if (gf2_rank_64() == 64) {
      matrix_loaded = true;
      if (attempts_used) *attempts_used = attempt + 1;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Independent verifier: second Keccak with constants derived from the spec
// (LFSR round constants, rho offsets from the (x,y) walk), lane-wise state,
// separate cSHAKE padding and a plain matrix-vector product.
// ---------------------------------------------------------------------------

static uint64_t ref_rc[24];
static int ref_rho[25];

static inline uint64_t ref_rotl(uint64_t x, int n) { n &= 63; return n ? ((x << n) | (x >> (64 - n))) : x; }

static bool ref_lfsr_bit(int t) {
  uint16_t R = 1;
  for (int i = 1; i <= t % 255; i++) {
    R <<= 1;
    if (R & 0x100) R ^= 0x171;
  }
  return R & 1;
}

static void ref_init() {
  for (int i = 0; i < 24; i++) {
    uint64_t c = 0;
    for (int j = 0; j < 7; j++) if (ref_lfsr_bit(j + 7 * i)) c |= (1ULL << ((1 << j) - 1));
    ref_rc[i] = c;
  }
  for (int i = 0; i < 25; i++) ref_rho[i] = 0;
  int x = 1, y = 0;
  for (int t = 0; t < 24; t++) {
    ref_rho[x + 5 * y] = ((t + 1) * (t + 2) / 2) % 64;
    int nx = y, ny = (2 * x + 3 * y) % 5;
    x = nx; y = ny;
  }
}

static void ref_keccakf(uint64_t A[25]) {
  uint64_t B[25], C[5], D[5];
  for (int r = 0; r < 24; r++) {
    for (int x = 0; x < 5; x++) C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
    for (int x = 0; x < 5; x++) D[x] = C[(x + 4) % 5] ^ ref_rotl(C[(x + 1) % 5], 1);
    for (int i = 0; i < 25; i++) A[i] ^= D[i % 5];
    for (int x = 0; x < 5; x++)
      for (int y = 0; y < 5; y++)
        B[y + 5 * ((2 * x + 3 * y) % 5)] = ref_rotl(A[x + 5 * y], ref_rho[x + 5 * y]);
    for (int x = 0; x < 5; x++)
      for (int y = 0; y < 5; y++)
        A[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
    A[0] ^= ref_rc[r];
  }
}

static uint64_t ref_load64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

// out: 32 bytes. datalen <= 80.
static void ref_cshake256_32(const uint8_t *data, size_t datalen, const char *custom, uint8_t out[32]) {
  uint8_t msg[RATE * 2 + 8];
  memset(msg, 0, sizeof(msg));
  size_t clen = strlen(custom), p = 0;
  msg[p++] = 1; msg[p++] = RATE;                       // left_encode(136)
  msg[p++] = 1; msg[p++] = 0;                          // encode_string("")
  msg[p++] = 1; msg[p++] = (uint8_t)(clen * 8);        // left_encode(bits), clen*8 < 256
  memcpy(msg + p, custom, clen);
  size_t L = RATE + datalen;                           // prefix is exactly one block
  memcpy(msg + RATE, data, datalen);
  size_t padded = ((L / RATE) + 1) * RATE;
  msg[L] ^= 0x04;
  msg[padded - 1] ^= 0x80;

  uint64_t A[25];
  memset(A, 0, sizeof(A));
  for (size_t off = 0; off < padded; off += RATE) {
    for (int i = 0; i < RATE / 8; i++) A[i] ^= ref_load64(msg + off + 8 * i);
    ref_keccakf(A);
  }
  for (int i = 0; i < 4; i++) store64_le(out + 8 * i, A[i]);
}

static void ref_kheavy_hash(uint64_t nonce, uint8_t out[32]) {
  uint8_t work[48], pow_hash[32], transformed[32], fin[32];
  memcpy(work, pre_pow, 32);
  for (int i = 0; i < 8; i++) work[32 + i] = (timestamp >> (8 * i)) & 0xff;
  for (int i = 0; i < 8; i++) work[40 + i] = (nonce >> (8 * i)) & 0xff;
  ref_cshake256_32(work, 48, "ProofOfWorkHash", pow_hash);

  uint32_t prod[64];
  for (int i = 0; i < 64; i++) {
    uint32_t s = 0;
    for (int j = 0; j < 64; j++) {
      uint32_t v = (j & 1) ? (pow_hash[j / 2] & 0x0f) : (pow_hash[j / 2] >> 4);
      s += (uint32_t)matrix[i][j] * v;
    }
    prod[i] = s >> 10;
  }
  for (int i = 0; i < 32; i++)
    transformed[i] = pow_hash[i] ^ (uint8_t)(((prod[2 * i] & 0xf) << 4) | (prod[2 * i + 1] & 0xf));

  ref_cshake256_32(transformed, 32, "HeavyHash", fin);
  for (int i = 0; i < 32; i++) out[i] = fin[31 - i];
}

// ---------------------------------------------------------------------------
// Hex / print helpers
// ---------------------------------------------------------------------------

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool hex_to_bytes(const char *s, uint8_t *out, size_t n) {
  if (!s || strlen(s) != n * 2) return false;
  for (size_t i = 0; i < n; i++) {
    int hi = hexval(s[2 * i]), lo = hexval(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

static uint64_t parse_hex64(const char *s) {
  uint64_t v = 0;
  for (int i = 0; s && s[i] && i < 16; i++) {
    int h = hexval(s[i]);
    if (h < 0) break;
    v = (v << 4) | (uint64_t)h;
  }
  return v;
}

static uint64_t parse_u64(const char *s) {  // decimal, or 0x-prefixed hex
  if (!s) return 0;
  if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return parse_hex64(s + 2);
  uint64_t v = 0;
  while (*s >= '0' && *s <= '9') v = v * 10 + (uint64_t)(*s++ - '0');
  return v;
}

static void print_hex64(uint64_t v) {
  for (int i = 15; i >= 0; i--) {
    uint8_t nib = (v >> (i * 4)) & 0xF;
    Serial.print((char)(nib < 10 ? '0' + nib : 'a' + nib - 10));
  }
}

static void print_hex_bytes(const uint8_t *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (b[i] < 16) Serial.print('0');
    Serial.print(b[i], HEX);
  }
}

static void print_u64_dec(uint64_t v) {
  char buf[24];
  int n = 0;
  if (v == 0) buf[n++] = '0';
  while (v > 0) { buf[n++] = '0' + (int)(v % 10); v /= 10; }
  while (n > 0) Serial.print(buf[--n]);
}

// ---------------------------------------------------------------------------
// Job helpers (formerly PC side)
// ---------------------------------------------------------------------------

static void target_from_leading_zero_bits(int bits) {
  if (bits < 1) bits = 1;
  if (bits > 255) bits = 255;
  for (int i = 0; i < 32; i++) {
    int lo = i * 8;
    if (bits >= lo + 8) target[i] = 0x00;
    else if (bits <= lo) target[i] = 0xff;
    else target[i] = 0xff >> (bits - lo);
  }
}

static void load_demo_template() {
  const char *d = "kaspa-optical-bench-demo-template";
  cshake256((const uint8_t *)d, strlen(d), "ProofOfWorkHash", pre_pow, 32);
}

static void init_hash_states() {
  cshake_init(st_pow0, "ProofOfWorkHash");
  cshake_init(st_heavy0, "HeavyHash");
  ref_init();
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

static void read_telemetry(uint16_t *vals, int *n) {
  if (source == SRC_CAMERA) {
    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++) vals[y * 8 + x] = sample_pd(x) >> 2;
    *n = 64;
  } else if (source == SRC_PHOTODIODES) {
    for (int i = 0; i < NUM_PD; i++) vals[i] = sample_pd(i);
    *n = NUM_PD;
  } else {
    *n = 0;
  }
}

static void send_sample() {
  uint16_t v[64];
  int n;
  read_telemetry(v, &n);
  if (n == 0) { Serial.println(F("ERR,SOURCE_NONE")); return; }
  Serial.print(source == SRC_CAMERA ? F("F,") : F("D,"));
  Serial.print(millis());
  for (int i = 0; i < n; i++) { Serial.print(','); Serial.print(v[i]); }
  Serial.println();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Returns 1 found, 0 exhausted, -1 stopped by user. *tried = nonces hashed.
static int search(uint64_t first, uint64_t count, uint64_t *found_nonce, uint8_t found_hash[32], uint64_t *tried, bool interactive) {
  uint8_t h[32];
  *tried = 0;
  for (uint64_t k = 0; k < count; k++) {
    uint64_t nonce = first + k;
    kheavy_hash(nonce, h);
    (*tried)++;

    if (ito_every && (k % ito_every) == 0) update_optics(h);

    if (hash_meets_target(h)) {
      *found_nonce = nonce;
      memcpy(found_hash, h, 32);
      update_optics(h);
      return 1;
    }

    if (interactive && (k % batch_size) == (batch_size - 1)) {
      if (Serial.available()) {
        while (Serial.available()) Serial.read();
        return -1;
      }
    }
  }
  return 0;
}

static void report_json(bool solved, uint64_t nonce, const uint8_t *hash, bool verified,
                        uint64_t tried, float elapsed_s, const uint8_t *ito,
                        const uint16_t *tel, int tel_n, int matrix_attempts) {
  Serial.print(F("REPORT,{\"solved\":")); Serial.print(solved ? F("true") : F("false"));
  Serial.print(F(",\"nonces_attempted\":")); print_u64_dec(tried);
  Serial.print(F(",\"elapsed_s\":")); Serial.print(elapsed_s, 3);
  Serial.print(F(",\"bench_hashrate_approx\":")); Serial.print(elapsed_s > 0 ? (float)tried / elapsed_s : 0.0f, 1);
  Serial.print(F(",\"source\":\""));
  Serial.print(source == SRC_CAMERA ? F("camera") : source == SRC_PHOTODIODES ? F("photodiodes") : F("none"));
  Serial.print(F("\",\"timestamp\":")); print_u64_dec(timestamp);
  Serial.print(F(",\"pre_pow_hash_hex\":\"")); print_hex_bytes(pre_pow, 32);
  Serial.print(F("\",\"target_hex\":\"")); print_hex_bytes(target, 32);
  Serial.print(F("\",\"matrix_attempts\":")); Serial.print(matrix_attempts);
  if (solved) {
    Serial.print(F(",\"solution\":{\"nonce\":")); print_u64_dec(nonce);
    Serial.print(F(",\"nonce_hex\":\"")); print_hex64(nonce);
    Serial.print(F("\",\"hash_hex\":\"")); print_hex_bytes(hash, 32);
    Serial.print(F("\",\"verified_by_independent_check\":")); Serial.print(verified ? F("true") : F("false"));
    Serial.print(F(",\"ito_pattern\":["));
    for (int i = 0; i < NUM_ITO; i++) { if (i) Serial.print(','); Serial.print(ito[i]); }
    Serial.print(F("],\"telemetry\":["));
    for (int i = 0; i < tel_n; i++) { if (i) Serial.print(','); Serial.print(tel[i]); }
    Serial.print(F("]}"));
  }
  Serial.println(F("}"));
}

static void run_full() {
  int attempts = 0;
  if (!generate_matrix(1000, &attempts)) { Serial.println(F("ERR,MATRIX_GEN")); return; }
  Serial.print(F("OK,MATRIX_GENERATED,")); Serial.println(attempts);

  clear_optics();
  uint64_t tried_total = 0, nonce = 0;
  uint8_t hash[32];
  int res = 0;
  uint32_t t0 = millis();

  while (tried_total < max_nonces) {
    uint64_t chunk = max_nonces - tried_total;
    if (chunk > 4096) chunk = 4096;
    uint64_t tried = 0;
    res = search(start_nonce + tried_total, chunk, &nonce, hash, &tried, true);
    tried_total += tried;
    if (res != 0) break;
    Serial.print(F("OK,PROGRESS,")); print_u64_dec(tried_total); Serial.println();
  }

  float elapsed = (millis() - t0) / 1000.0f;

  if (res == 1) {
    uint8_t ref[32];
    ref_kheavy_hash(nonce, ref);
    bool verified = memcmp(ref, hash, 32) == 0 && hash_meets_target(ref);

    uint8_t ito[NUM_ITO];
    for (int i = 0; i < NUM_ITO; i++) ito[i] = hash[i];
    update_optics(hash);
    delay(5);  // let the pixels settle before sampling

    uint16_t tel[64];
    int tel_n = 0;
    read_telemetry(tel, &tel_n);

    Serial.print(F("FOUND,")); print_hex64(nonce); Serial.print(',');
    print_hex_bytes(hash, 32); Serial.println();

    Serial.print(F("CSV,")); print_hex64(nonce); Serial.print(',');
    print_hex_bytes(hash, 32); Serial.print(',');
    for (int i = 0; i < NUM_ITO; i++) { if (i) Serial.print('/'); Serial.print(ito[i]); }
    Serial.print(','); Serial.print(millis());
    for (int i = 0; i < tel_n; i++) { Serial.print(','); Serial.print(tel[i]); }
    Serial.println();

    report_json(true, nonce, hash, verified, tried_total, elapsed, ito, tel, tel_n, attempts);
  } else {
    if (res == -1) Serial.println(F("OK,STOPPED"));
    else { Serial.print(F("NONE,")); print_hex64(tried_total); Serial.println(); }
    report_json(false, 0, NULL, false, tried_total, elapsed, NULL, NULL, 0, attempts);
    clear_optics();
  }
}

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------

static char *next_tok(char **p) {
  char *s = *p;
  while (*s == ' ') s++;
  if (!*s) { *p = s; return NULL; }
  char *start = s;
  while (*s && *s != ' ') s++;
  if (*s) *s++ = 0;
  *p = s;
  return start;
}

static void handle_line(char *line) {
  char *rest = line;
  char *cmd = next_tok(&rest);
  if (!cmd) return;
  for (char *c = cmd; *c; c++) if (*c >= 'a' && *c <= 'z') *c -= 32;

  if (!strcmp(cmd, "PING")) {
    Serial.println(F("OK,PONG"));

  } else if (!strcmp(cmd, "MODE")) {
    char *m = next_tok(&rest);
    if (!m) { Serial.println(F("ERR,MODE syntax")); return; }
    if (!strcasecmp(m, "PHOTODIODES")) { source = SRC_PHOTODIODES; Serial.println(F("OK,MODE_PHOTODIODES")); }
    else if (!strcasecmp(m, "CAMERA")) { source = SRC_CAMERA; Serial.println(F("OK,MODE_CAMERA")); }
    else if (!strcasecmp(m, "NONE")) { source = SRC_NONE; Serial.println(F("OK,MODE_NONE")); }
    else Serial.println(F("ERR,MODE syntax"));

  } else if (!strcmp(cmd, "SAMPLE")) {
    send_sample();

  } else if (!strcmp(cmd, "ITOCLEAR")) {
    clear_optics();
    Serial.println(F("OK,ITOCLEAR"));

  } else if (!strcmp(cmd, "ITOEVERY")) {
    uint64_t n = parse_u64(next_tok(&rest));
    ito_every = n ? (uint32_t)n : 1;
    Serial.println(F("OK,ITOEVERY"));

  } else if (!strcmp(cmd, "SETITO")) {
    int vals[NUM_ITO];
    for (int i = 0; i < NUM_ITO; i++) {
      char *t = next_tok(&rest);
      if (!t) { Serial.println(F("ERR,SETITO syntax")); return; }
      for (char *c = t; *c; c++) if (*c < '0' || *c > '9') { Serial.println(F("ERR,SETITO syntax")); return; }
      vals[i] = constrain((int)parse_u64(t), 0, 255);
    }
    if (next_tok(&rest)) { Serial.println(F("ERR,SETITO syntax")); return; }
    for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], vals[i]);
    Serial.println(F("OK,SETITO"));

  } else if (!strcmp(cmd, "MATRIXROW")) {
    char *r = next_tok(&rest);
    char *hex = next_tok(&rest);
    if (!r || !hex) { Serial.println(F("ERR,MATRIXROW syntax")); return; }
    int row = (int)parse_u64(r);
    if (row < 0 || row >= 64) { Serial.println(F("ERR,MATRIXROW index")); return; }
    if (!hex_to_bytes(hex, matrix[row], 64)) { Serial.println(F("ERR,MATRIXROW hex")); return; }
    matrix_loaded = true;
    Serial.println(F("OK,MATRIXROW"));

  } else if (!strcmp(cmd, "GENMATRIX")) {
    int attempts = 0;
    if (generate_matrix(1000, &attempts)) { Serial.print(F("OK,MATRIX_GENERATED,")); Serial.println(attempts); }
    else Serial.println(F("ERR,MATRIX_GEN"));

  } else if (!strcmp(cmd, "PREPOW")) {
    if (!hex_to_bytes(next_tok(&rest), pre_pow, 32)) { Serial.println(F("ERR,PREPOW hex")); return; }
    matrix_loaded = false;
    Serial.println(F("OK,PREPOW"));

  } else if (!strcmp(cmd, "WORKORDER")) {
    uint8_t work[80];
    if (!hex_to_bytes(next_tok(&rest), work, 80)) { Serial.println(F("ERR,WORKORDER hex")); return; }
    cshake256(work, 80, "ProofOfWorkHash", pre_pow, 32);
    matrix_loaded = false;
    Serial.println(F("OK,WORKORDER"));

  } else if (!strcmp(cmd, "DEMO")) {
    load_demo_template();
    matrix_loaded = false;
    Serial.println(F("OK,DEMO"));

  } else if (!strcmp(cmd, "TIMESTAMP")) {
    timestamp = parse_u64(next_tok(&rest));
    Serial.println(F("OK,TIMESTAMP"));

  } else if (!strcmp(cmd, "TARGETBITS")) {
    target_from_leading_zero_bits((int)parse_u64(next_tok(&rest)));
    Serial.println(F("OK,TARGETBITS"));

  } else if (!strcmp(cmd, "TARGET")) {
    if (!hex_to_bytes(next_tok(&rest), target, 32)) { Serial.println(F("ERR,TARGET hex")); return; }
    Serial.println(F("OK,TARGET"));

  } else if (!strcmp(cmd, "START")) {
    start_nonce = parse_u64(next_tok(&rest));
    Serial.println(F("OK,START"));

  } else if (!strcmp(cmd, "MAXNONCES")) {
    max_nonces = parse_u64(next_tok(&rest));
    Serial.println(F("OK,MAXNONCES"));

  } else if (!strcmp(cmd, "BATCH")) {
    uint64_t n = parse_u64(next_tok(&rest));
    batch_size = n ? (uint32_t)n : 1;
    Serial.println(F("OK,BATCH"));

  } else if (!strcmp(cmd, "RUN")) {
    run_full();

  } else if (!strcmp(cmd, "JOB")) {
    // Legacy: JOB <pre_pow> <timestamp> <target> <start_nonce> <count>; uses the loaded matrix.
    if (!matrix_loaded) { Serial.println(F("ERR,MATRIX_NOT_LOADED")); return; }
    char *a = next_tok(&rest), *b = next_tok(&rest), *c = next_tok(&rest);
    char *d = next_tok(&rest), *e = next_tok(&rest);
    if (!a || !b || !c || !d || !e) { Serial.println(F("ERR,JOB syntax")); return; }
    if (!hex_to_bytes(a, pre_pow, 32) || !hex_to_bytes(c, target, 32)) { Serial.println(F("ERR,JOB hex")); return; }
    timestamp = parse_hex64(b);
    uint64_t first = parse_hex64(d), count = parse_hex64(e);

    uint64_t nonce = 0, tried = 0;
    uint8_t hash[32];
    if (search(first, count, &nonce, hash, &tried, false) == 1) {
      Serial.print(F("FOUND,")); print_hex64(nonce); Serial.print(',');
      print_hex_bytes(hash, 32); Serial.println();
    } else {
      Serial.print(F("NONE,")); print_hex64(count); Serial.println();
    }

  } else {
    Serial.println(F("ERR,UNKNOWN_COMMAND"));
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  for (int i = 0; i < 3; ++i) {
    pinMode(MUX_SELECT_PINS[i], OUTPUT);
    digitalWrite(MUX_SELECT_PINS[i], LOW);
  }
  pinMode(PHOTO_ADC_PIN, INPUT);
  for (int i = 0; i < NUM_ITO; i++) {
    pinMode(ITO_PINS[i], OUTPUT);
    analogWrite(ITO_PINS[i], 0);
  }

  init_hash_states();
  load_demo_template();
  target_from_leading_zero_bits(12);  // easy bench target

  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 4000) delay(10);

  Serial.println(F("OK,READY"));

#if AUTO_RUN_ON_BOOT
  Serial.println(F("OK,AUTORUN_DEMO"));
  run_full();
#endif
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line_buf[line_len] = 0;
      handle_line(line_buf);
      line_len = 0;
    } else if (line_len < sizeof(line_buf) - 1) {
      line_buf[line_len++] = c;
    } else {
      line_len = 0;
      Serial.println(F("ERR,LINE_TOO_LONG"));
    }
  }
}
