
// Arduino Mega 2560 - fully offloaded Kaspa kHeavyHash bench miner.
//
// Photodiodes: A0..A7
// ITO pixels:  D2..D9, PWM 0..255
//
// PC -> Mega:
//   PING
//   MODE PHOTODIODES | MODE CAMERA
//   MATRIXROW <row 0..63> <128 hex chars: 64 4-bit matrix entries>
//   ITOEVERY <n>
//   JOB <pre_pow_hash 64 hex> <timestamp 16 hex> <target 64 hex> <start_nonce 16 hex> <count 16 hex>
//   SAMPLE
//   ITOCLEAR
//
// Mega -> PC:
//   OK,...
//   ERR,...
//   D,<millis>,<pd0..pd7>
//   F,<millis>,<64 camera values>
//   FOUND,<nonce 16 hex>,<hash 64 hex>
//   NONE,<nonces tried 16 hex>
//
// Optical readings are telemetry only. They do not validate proof-of-work.

#define RATE 136
#define NUM_ITO 8
#define NUM_PD 8

static const int PHOTO_PINS[NUM_PD] = {A0, A1, A2, A3, A4, A5, A6, A7};
static const int ITO_PINS[NUM_ITO] = {2, 3, 4, 5, 6, 7, 8, 9};

static uint8_t matrix[64][64];
static bool matrix_loaded = false;
static bool camera_mode = false;
static uint32_t ito_every = 1000;

// ---------------------------------------------------------------------------
// Keccak-f[1600]
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
    for (int i = 0; i < 5; i++) {
      bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    }

    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
      for (int j = 0; j < 25; j += 5) {
        st[j + i] ^= t;
      }
    }

    t = st[1];
    for (int i = 0; i < 24; i++) {
      int j = PILN[i];
      bc[0] = st[j];
      st[j] = ROL64(t, ROTC[i]);
      t = bc[0];
    }

    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; i++) {
        bc[i] = st[j + i];
      }
      for (int i = 0; i < 5; i++) {
        st[j + i] ^= bc[(i + 1) % 5] ^ ((~bc[(i + 2) % 5]) & bc[(i + 3) % 5]);
      }
    }

    st[0] ^= RC[round];
  }
}

// ---------------------------------------------------------------------------
// cSHAKE256
// ---------------------------------------------------------------------------

static uint64_t load64_le(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) {
    v = (v << 8) | p[i];
  }
  return v;
}

static void store64_le(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    p[i] = (v >> (8 * i)) & 0xff;
  }
}

static void xor_block(uint64_t st[25], const uint8_t *block, size_t len) {
  uint8_t *state_bytes = (uint8_t *)st;
  for (size_t i = 0; i < len; i++) {
    state_bytes[i] ^= block[i];
  }
}

static void keccak_absorb(uint64_t st[25], const uint8_t *data, size_t len) {
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
    if (chunk > RATE) {
      chunk = RATE;
    }

    memcpy(out + made, (uint8_t *)st, chunk);
    made += chunk;

    if (made < outlen) {
      keccakf(st);
    }
  }
}

static size_t left_encode(uint8_t *out, size_t x) {
  uint8_t tmp[8];
  size_t n = 0;

  while (x > 0) {
    tmp[n++] = x & 0xff;
    x >>= 8;
  }

  if (n == 0) {
    out[0] = 1;
    out[1] = 0;
    return 2;
  }

  out[0] = n;
  for (size_t i = 0; i < n; i++) {
    out[1 + i] = tmp[n - 1 - i];
  }

  return n + 1;
}

static size_t encode_string(uint8_t *out, const char *s) {
  size_t len = strlen(s);
  size_t n = left_encode(out, len * 8);
  memcpy(out + n, s, len);
  return n + len;
}

static void cshake256(
  const uint8_t *data,
  size_t datalen,
  const char *custom,
  uint8_t *out,
  size_t outlen
) {
  uint64_t st[25];
  memset(st, 0, sizeof(st));

  uint8_t prefix[RATE];
  memset(prefix, 0, RATE);

  size_t p = 0;
  p += left_encode(prefix + p, RATE);
  p += left_encode(prefix + p, 0);          // Empty function-name N.
  p += encode_string(prefix + p, custom);   // Customization string S.

  keccak_absorb(st, prefix, RATE);
  keccak_absorb(st, data, datalen);
  keccak_squeeze(st, out, outlen);
}

// ---------------------------------------------------------------------------
// kHeavyHash
// ---------------------------------------------------------------------------

static uint8_t pre_pow[32];
static uint8_t target[32];
static uint64_t timestamp;
static uint64_t start_nonce;

static void heavy_transform(const uint8_t pow_hash[32], uint8_t out[32]) {
  uint16_t vector[64];

  for (int i = 0; i < 32; i++) {
    vector[2 * i] = pow_hash[i] >> 4;
    vector[2 * i + 1] = pow_hash[i] & 0x0f;
  }

  uint8_t product[32];

  for (int i = 0; i < 64; i += 2) {
    uint32_t a = 0;
    uint32_t b = 0;

    for (int j = 0; j < 64; j++) {
      a += (uint32_t)matrix[i][j] * vector[j];
      b += (uint32_t)matrix[i + 1][j] * vector[j];
    }

    product[i / 2] = (uint8_t)(((a >> 10) << 4) | ((b >> 10) & 0x0f));
  }

  for (int i = 0; i < 32; i++) {
    out[i] = pow_hash[i] ^ product[i];
  }
}

static bool hash_meets_target(const uint8_t h[32]) {
  for (int i = 0; i < 32; i++) {
    if (h[i] < target[i]) return true;
    if (h[i] > target[i]) return false;
  }
  return true;
}

static void update_optics(const uint8_t h[32]) {
  for (int i = 0; i < NUM_ITO; i++) {
    analogWrite(ITO_PINS[i], h[i]);
  }
}

static bool mine(uint64_t count, uint64_t &found_nonce, uint8_t found_hash[32]) {
  uint8_t work[48];
  uint8_t pow_hash[32];
  uint8_t transformed[32];
  uint8_t final_hash[32];

  memcpy(work, pre_pow, 32);
  store64_le(work + 32, timestamp);

  for (uint64_t k = 0; k < count; k++) {
    uint64_t nonce = start_nonce + k;
    store64_le(work + 40, nonce);

    cshake256(work, 48, "ProofOfWorkHash", pow_hash, 32);
    heavy_transform(pow_hash, transformed);
    cshake256(transformed, 32, "HeavyHash", final_hash, 32);

    // Kaspa compares the byte-reversed final hash.
    for (int i = 0; i < 16; i++) {
      uint8_t tmp = final_hash[i];
      final_hash[i] = final_hash[31 - i];
      final_hash[31 - i] = tmp;
    }

    if (k % ito_every == 0) {
      update_optics(final_hash);
    }

    if (hash_meets_target(final_hash)) {
      found_nonce = nonce;
      memcpy(found_hash, final_hash, 32);
      update_optics(final_hash);
      return true;
    }
  }

  return false;
}

// ---------------------------------------------------------------------------
// Serial helpers
// ---------------------------------------------------------------------------

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool hex_to_bytes(const String &s, uint8_t *out, size_t n) {
  if (s.length() != (int)(n * 2)) return false;

  for (size_t i = 0; i < n; i++) {
    int hi = hexval(s[2 * i]);
    int lo = hexval(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }

  return true;
}

static uint64_t parse_hex64(const String &s) {
  uint64_t v = 0;

  for (unsigned int i = 0; i < s.length() && i < 16; i++) {
    int h = hexval(s[i]);
    if (h < 0) break;
    v = (v << 4) | (uint64_t)h;
  }

  return v;
}

static void print_hex64(uint64_t v) {
  for (int i = 15; i >= 0; i--) {
    uint8_t nib = (v >> (i * 4)) & 0xF;
    Serial.print(nib < 10 ? (char)('0' + nib) : (char)('A' + nib - 10));
  }
}

static void send_photodiodes() {
  Serial.print(F("D,"));
  Serial.print(millis());

  for (int i = 0; i < NUM_PD; i++) {
    Serial.print(',');
    Serial.print(analogRead(PHOTO_PINS[i]));
  }

  Serial.println();
}

static void send_camera() {
  // Default implementation: tile A0..A7 into an 8x8 frame.
  // Replace this with your actual camera-module capture code.
  uint8_t frame[64];

  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      frame[y * 8 + x] = analogRead(PHOTO_PINS[x]) >> 2;
    }
  }

  Serial.print(F("F,"));
  Serial.print(millis());

  for (int i = 0; i < 64; i++) {
    Serial.print(',');
    Serial.print(frame[i]);
  }

  Serial.println();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < NUM_ITO; i++) {
    pinMode(ITO_PINS[i], OUTPUT);
    analogWrite(ITO_PINS[i], 0);
  }

  Serial.println(F("OK,READY"));
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line == F("PING")) {
    Serial.println(F("OK,PONG"));

  } else if (line == F("MODE PHOTODIODES")) {
    camera_mode = false;
    Serial.println(F("OK,MODE_PHOTODIODES"));

  } else if (line == F("MODE CAMERA")) {
    camera_mode = true;
    Serial.println(F("OK,MODE_CAMERA"));

  } else if (line == F("SAMPLE")) {
    if (camera_mode) {
      send_camera();
    } else {
      send_photodiodes();
    }

  } else if (line == F("ITOCLEAR")) {
    for (int i = 0; i < NUM_ITO; i++) {
      analogWrite(ITO_PINS[i], 0);
    }
    Serial.println(F("OK,ITOCLEAR"));

  } else if (line.startsWith(F("ITOEVERY "))) {
    uint32_t n = (uint32_t)line.substring(9).toInt();
    if (n == 0) n = 1;
    ito_every = n;
    Serial.println(F("OK,ITOEVERY"));

  } else if (line.startsWith(F("MATRIXROW "))) {
    int sp = line.indexOf(' ', 10);
    if (sp < 0) {
      Serial.println(F("ERR,MATRIXROW syntax"));
      return;
    }

    int row = line.substring(10, sp).toInt();
    if (row < 0 || row >= 64) {
      Serial.println(F("ERR,MATRIXROW index"));
      return;
    }

    if (!hex_to_bytes(line.substring(sp + 1), matrix[row], 64)) {
      Serial.println(F("ERR,MATRIXROW hex"));
      return;
    }

    matrix_loaded = true;
    Serial.println(F("OK,MATRIXROW"));

  } else if (line.startsWith(F("SETITO "))) {
    int values[NUM_ITO];
    int pos = 7;
    bool valid = true;
    for (int i = 0; i < NUM_ITO; i++) {
      while (pos < (int)line.length() && line.charAt(pos) == ' ') pos++;
      if (pos >= (int)line.length()) {
        valid = false;
        break;
      }
      int end = line.indexOf(' ', pos);
      if (end < 0) end = line.length();
      String token = line.substring(pos, end);
      for (unsigned int j = 0; j < token.length(); j++) {
        if (!isDigit(token.charAt(j))) valid = false;
      }
      if (!valid || token.length() == 0) {
        valid = false;
        break;
      }
      values[i] = constrain(token.toInt(), 0, 255);
      pos = end + 1;
    }
    while (pos < (int)line.length() && line.charAt(pos) == ' ') pos++;
    if (pos < (int)line.length()) valid = false;
    if (!valid) {
      Serial.println(F("ERR,SETITO syntax"));
      return;
    }
    for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], values[i]);
    Serial.println(F("OK,SETITO"));

  } else if (line.startsWith(F("JOB "))) {
    if (!matrix_loaded) {
      Serial.println(F("ERR,MATRIX_NOT_LOADED"));
      return;
    }

    // JOB has exactly five fields after the command:
    // pre-PoW hash, timestamp, target, start nonce, count.
    int p1 = line.indexOf(' ', 4);
    int p2 = (p1 >= 0) ? line.indexOf(' ', p1 + 1) : -1;
    int p3 = (p2 >= 0) ? line.indexOf(' ', p2 + 1) : -1;
    int p4 = (p3 >= 0) ? line.indexOf(' ', p3 + 1) : -1;

    if (p1 < 0 || p2 < 0 || p3 < 0 || p4 < 0) {
      Serial.println(F("ERR,JOB syntax"));
      return;
    }

    if (!hex_to_bytes(line.substring(4, p1), pre_pow, 32) ||
        !hex_to_bytes(line.substring(p2 + 1, p3), target, 32)) {
      Serial.println(F("ERR,JOB hex"));
      return;
    }

    timestamp = parse_hex64(line.substring(p1 + 1, p2));
    start_nonce = parse_hex64(line.substring(p3 + 1, p4));
    uint64_t count = parse_hex64(line.substring(p4 + 1));

    uint64_t found_nonce = 0;
    uint8_t found_hash[32];

    if (mine(count, found_nonce, found_hash)) {
      Serial.print(F("FOUND,"));
      print_hex64(found_nonce);
      Serial.print(',');

      for (int i = 0; i < 32; i++) {
        if (found_hash[i] < 16) Serial.print('0');
        Serial.print(found_hash[i], HEX);
      }

      Serial.println();
    } else {
      Serial.print(F("NONE,"));
      print_hex64(count);
      Serial.println();
    }

  } else {
    Serial.println(F("ERR,UNKNOWN_COMMAND"));
  }
}
