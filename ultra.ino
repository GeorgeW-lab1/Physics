// ============================================================================
// Cloud-Compatible Kaspa Photonic Mining Engine (Autonomous Nonce Loop)
// Architecture: Universal AVR 8-bit Framework for Differential Optical Core
// ============================================================================

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Master Cryptographic Mining Variables
static uint64_t s_current_nonce = 0;        // 64-bit mining nonce iterator
static uint8_t  s_pow_hash[32];              // Captured intermediate Pre-PoW hash 
static uint8_t  s_final_hash[32];            // Final processed hash sent to target check
static bool     s_block_found = false;       // Core mining execution state flag

// Mining Difficulty Target
// Lower target values mean higher network difficulty.
// 4 leading hex zeros (16 leading zero bits) makes it ideal for local testing.
static const uint8_t MINING_DIFFICULTY_TARGET[32] = {
    0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

// Physical Wire Mapping on your Bench Layout:
// Pins 22 to 29 -> Outputs to your 8 Pixelated ITO Mask Channels
// Pins 30 to 37 -> Inputs from your parallel quantized Photodiodes
// Pin 11        -> IR Strobe A Control (Normal Bit Path)
// Pin 12        -> IR Strobe B Control (Inverted Bit Path)
const uint8_t PIN_ITO_START = 22;
const uint8_t PIN_PD_START  = 30;
const uint8_t PIN_IR_STROBE_A = 11;
const uint8_t PIN_IR_STROBE_B = 12;

// Authentic 80-byte Kaspa Header Work Order Vector Template
static uint8_t s_kaspa_workorder[80] = {
    0x01, 0x00, 0x00, 0x00, 0x5a, 0x2e, 0x93, 0xc1, 0x1f, 0x44, 0x8a, 0xde, 0xbc, 0x33, 0x22, 0x11,
    0x00, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x64, 0x9e, 0xbc, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// ============================================================================
// CRYPTOGRAPHIC PIPELINE: SHA-3 / Keccak-f Core Implementation
// ============================================================================

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000080008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int KECCAK_PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

static const int KECCAK_ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static void keccakf1600(uint64_t* st) {
    uint64_t t, bc[5];
    for (int round = 0; round < 24; round++) {
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PILN[i];
            bc[0] = st[j];
            st[j] = ROL64(t, KECCAK_ROTC[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= KECCAK_RC[round];
    }
}

// Stage 1: Computes cSHAKE256 ("ProofOfWorkHash") over Header + Nonce
void run_pre_pow_hash(uint64_t nonce) {
    uint64_t st[25];
    memset(st, 0, sizeof(st));

    // Latch current nonce into the trailing bytes of the work order buffer (bytes 72-79)
    memcpy(&s_kaspa_workorder[72], &nonce, sizeof(nonce));

    // NIST SP 800-185 Prefix Setup
    uint8_t prefix[136] = {0};
    prefix[0] = 0x01; prefix[1] = 0x88; 
    prefix[2] = 0x01; prefix[3] = 0x00; 
    prefix[4] = 0x01; prefix[5] = 0x78; 
    memcpy(&prefix[6], "ProofOfWorkHash", 15);

    uint8_t* byte_st = (uint8_t*)st;
    for (int i = 0; i < 136; i++) byte_st[i] ^= prefix[i];
    keccakf1600(st);

    // Absorb Block Data Header Payload
    uint8_t msg_block[136] = {0};
    memcpy(msg_block, s_kaspa_workorder, 80);
    msg_block[80] ^= 0x04;
    msg_block[135] ^= 0x80;

    for (int i = 0; i < 136; i++) byte_st[i] ^= msg_block[i];
    keccakf1600(st);

    memcpy(s_pow_hash, byte_st, 32);
}

// Stage 3: Computes cSHAKE256 ("HeavyHash") over Optical Transformation output
void run_final_heavy_hash(const uint8_t* opt_matrix_out) {
    uint64_t st[25];
    memset(st, 0, sizeof(st));

    // NIST SP 800-185 Prefix Setup for "HeavyHash" Custom Domain String
    uint8_t prefix[136] = {0};
    prefix[0] = 0x01; prefix[1] = 0x88;
    prefix[2] = 0x01; prefix[3] = 0x00;
    prefix[4] = 0x01; prefix[5] = 0x48; 
    memcpy(&prefix[6], "HeavyHash", 9);

    uint8_t* byte_st = (uint8_t*)st;
    for (int i = 0; i < 136; i++) byte_st[i] ^= prefix[i];
    keccakf1600(st);

    // Absorb Optically Transformed Matrix Buffer
    uint8_t msg_block[136] = {0};
    memcpy(msg_block, opt_matrix_out, 32);
    msg_block[32] ^= 0x04;
    msg_block[135] ^= 0x80;

    for (int i = 0; i < 136; i++) byte_st[i] ^= msg_block[i];
    keccakf1600(st);

    // Output final candidate hash
    memcpy(s_final_hash, byte_st, 32);
}

// Custom internal printing logic to convert 64-bit variables into readable strings on 8-bit AVR compilers
void print_uint64(uint64_t value) {
    // If it fits inside a standard 32-bit unsigned long, print natively
    if (value <= 0xFFFFFFFFUL) {
        Serial.print((unsigned long)value);
    } else {
        // High-precision division routine to print massive 64-bit integer lines cleanly
        char buffer[21];
        uint8_t i = 0;
        while (value > 0) {
            buffer[i++] = '0' + (value % 10);
            value /= 10;
        }
        while (i > 0) {
            Serial.print(buffer[--i]);
        }
    }
}

// ============================================================================
// DUAL-STROBE DIFFERENTIAL PHOTONIC INTERFEROMETER ACCELERATOR CORE
// ============================================================================

void run_differential_optical_core(uint8_t* opt_out) {
    uint8_t* in_ptr = s_pow_hash;

    for (uint8_t i = 0; i < 32; i++) {
        uint8_t current_byte = in_ptr[i];

        // --- Sweep A: Direct Normal State Illumination ---
        digitalWrite(PIN_ITO_START + 0, (current_byte >> 0) & 0x01);
        digitalWrite(PIN_ITO_START + 1, (current_byte >> 1) & 0x01);
        digitalWrite(PIN_ITO_START + 2, (current_byte >> 2) & 0x01);
        digitalWrite(PIN_ITO_START + 3, (current_byte >> 3) & 0x01);
        digitalWrite(PIN_ITO_START + 4, (current_byte >> 4) & 0x01);
        digitalWrite(PIN_ITO_START + 5, (current_byte >> 5) & 0x01);
        digitalWrite(PIN_ITO_START + 6, (current_byte >> 6) & 0x01);
        digitalWrite(PIN_ITO_START + 7, (current_byte >> 7) & 0x01);

        digitalWrite(PIN_IR_STROBE_A, HIGH);
        delayMicroseconds(1); // Settle delay

        uint8_t sample_A = 0;
        for (uint8_t b = 0; b < 8; b++) {
            if (digitalRead(PIN_PD_START + b)) sample_A |= (1 << b);
        }
        digitalWrite(PIN_IR_STROBE_A, LOW);

        // --- Sweep B: Inverted State Balanced Signalling ---
        uint8_t inverted_byte = ~current_byte;
        digitalWrite(PIN_ITO_START + 0, (inverted_byte >> 0) & 0x01);
        digitalWrite(PIN_ITO_START + 1, (inverted_byte >> 1) & 0x01);
        digitalWrite(PIN_ITO_START + 2, (inverted_byte >> 2) & 0x01);
        digitalWrite(PIN_ITO_START + 3, (inverted_byte >> 3) & 0x01);
        digitalWrite(PIN_ITO_START + 4, (inverted_byte >> 4) & 0x01);
        digitalWrite(PIN_ITO_START + 5, (inverted_byte >> 5) & 0x01);
        digitalWrite(PIN_ITO_START + 6, (inverted_byte >> 6) & 0x01);
        digitalWrite(PIN_ITO_START + 7, (inverted_byte >> 7) & 0x01);

        digitalWrite(PIN_IR_STROBE_B, HIGH);
        delayMicroseconds(1);

        uint8_t sample_B = 0;
        for (uint8_t b = 0; b < 8; b++) {
            if (digitalRead(PIN_PD_START + b)) sample_B |= (1 << b);
        }
        digitalWrite(PIN_IR_STROBE_B, LOW);

        // --- Differential Common-Mode Rejection Resolution ---
        uint8_t matrix_reading = 0;
        for (uint8_t b = 0; b < 8; b++) {
            if ((sample_A & (1 << b)) && !(sample_B & (1 << b))) {
                matrix_reading |= (1 << b);
            }
        }

        // HeavyHash internal bitwise XOR mixing step
        opt_out[i] = current_byte ^ matrix_reading;
    }
}

// Evaluates the result vector against the active cryptographic network targets
bool check_hash_meets_target() {
    for (int i = 0; i < 32; i++) {
        if (s_final_hash[i] < MINING_DIFFICULTY_TARGET[i]) return true;
        if (s_final_hash[i] > MINING_DIFFICULTY_TARGET[i]) return false;
    }
    return true; 
}

// ============================================================================
// SETUP & EXECUTION CONTROL
// ============================================================================

void setup() {
    Serial.begin(115200);
while (!Serial) delay(10);
// Pin Configurations
pinMode(PIN_IR_STROBE_A, OUTPUT);
pinMode(PIN_IR_STROBE_B, OUTPUT);
digitalWrite(PIN_IR_STROBE_A, LOW);
digitalWrite(PIN_IR_STROBE_B, LOW);
for (uint8_t i = 0; i < 8; i++) {
pinMode(PIN_ITO_START + i, OUTPUT);
digitalWrite(PIN_ITO_START + i, LOW);
pinMode(PIN_PD_START + i, INPUT_PULLUP);
}
Serial.println(F("STATUS | Photonic Block Engine Online. Mining Init..."));
}
void loop() {
  // If a valid block hit was found, halt loop execution to prevent target corruption
  if (s_block_found) {
    delay(5000);
  return;
  }
  uint8_t optical_matrix_buffer[32];
  uint32_t batch_hashes = 20; // Lower batch size slightly for memory safety allocations
  unsigned long start_time = millis();
  for (uint32_t i = 0; i < batch_hashes; i++) {
  // Step 1: Compute intermediate cSHAKE256 pre-hash incorporating the unique active nonce
  run_pre_pow_hash(s_current_nonce);
  // Step 2: Push the pre-hash pattern out over the differential optical bench interface
  run_differential_optical_core(optical_matrix_buffer);
  // Step 3: Compute terminal cSHAKE256 final matrix-blend digest
  run_final_heavy_hash(optical_matrix_buffer);
  // Step 4: Validate hash validity boundaries against difficulty criteria
  if (check_hash_meets_target()) {
    s_block_found = true;
    Serial.println(F("\n=================================================="));
    Serial.println(F("!!! SUCCESS !!! BLOCK TARGET MET !!! CHIP HIT !!!"));
    Serial.print(F("Winning Nonce Found: "));
    print_uint64(s_current_nonce);
    Serial.println();
    Serial.print(F("Block Hash Header  : "));
    for (int h = 0; h < 32; h++) {
      if (s_final_hash[h] < 16) Serial.print('0');
        Serial.print(s_final_hash[h], HEX);
      }
        Serial.println(F("\n=================================================="));
      break;
    }
    s_current_nonce++; // Iterate nonce field to update state vectors for the next round
  }
  unsigned long end_time = millis();
  float seconds = (end_time - start_time) / 1000.0;
  if (!s_block_found && seconds > 0) {
  // Output live runtime hash rate status metrics using the custom uint64 converter
  Serial.print(F("Mining Telemetry | Current Nonce Index: "));
  print_uint64(s_current_nonce);
  Serial.print(F(" | Pipeline Hashrate: "));
  Serial.print((float)batch_hashes / seconds, 2);
  Serial.println(F(" H/s"));
  }
}
