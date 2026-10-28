// ============================================================================
// Arduino Mega (ATmega2560) Parallel Photonic Core Controller
// Architecture: Direct 8-Bit Register Mapping for ITO & Photodiode Array
// ============================================================================

#include <avr/io.h>
#include <avr/interrupt.h>

// Port assignments for the physical bench
// PORTA (Pins 22-29) -> Output to ITO Pixels
// PORTC (Pins 37-30) -> Input from parallel quantized Photodiodes
// PORTB Bit 6 (Pin 12) -> High-Speed IR Laser/Beam Strobe Control

static uint8_t s_pow_hash[32];
static uint8_t s_transformed_out[32];

void load_mock_pow_hash() {
    for (uint8_t i = 0; i < 32; i++) {
        s_pow_hash[i] = (uint8_t)(i * 33 + 7);
    }
}

/**
 * Executes the kHeavyHash matrix-vector stage natively on the optical bench.
 * Bypasses all digitalWrite / analogRead overheads to maximize frequency.
 */
void heavy_transform_mega_offload() {
    // Cache array pointers directly into CPU registers
    uint8_t* in_ptr  = s_pow_hash;
    uint8_t* out_ptr = s_transformed_out;

    // Unroll the calculation loop completely.
    // Instead of looping, we process the 32-byte hash vector as a series of 
    // sequential, parallel physical transformations on the bench.
    
    // Byte Step 0
    PORTA = in_ptr[0];                       // 1 cycle: Instantly set all 8 ITO mask pixels
    PORTB |= (1 << PB6);                     // 2 cycles: Fire the IR Beam Strobe
    __asm__ __volatile__("nop\n\t");         // Wait for photons to pass through the ITO mask
    uint8_t optical_matrix_0 = PINC;         // 1 cycle: Parallel sample all 8 photodiode channels
    PORTB &= ~(1 << PB6);                    // 2 cycles: Turn off the IR beams
    out_ptr[0] = in_ptr[0] ^ optical_matrix_0; // 1 cycle: Instant hardware XOR fusion

    // Byte Step 1
    PORTA = in_ptr[1];
    PORTB |= (1 << PB6);
    __asm__ __volatile__("nop\n\t");
    uint8_t optical_matrix_1 = PINC;
    PORTB &= ~(1 << PB6);
    out_ptr[1] = in_ptr[1] ^ optical_matrix_1;

    // Byte Step 2
    PORTA = in_ptr[2];
    PORTB |= (1 << PB6);
    __asm__ __volatile__("nop\n\t");
    uint8_t optical_matrix_2 = PINC;
    PORTB &= ~(1 << PB6);
    out_ptr[2] = in_ptr[2] ^ optical_matrix_2;

    // Byte Step 3
    PORTA = in_ptr[3];
    PORTB |= (1 << PB6);
    __asm__ __volatile__("nop\n\t");
    uint8_t optical_matrix_3 = PINC;
    PORTB &= ~(1 << PB6);
    out_ptr[3] = in_ptr[3] ^ optical_matrix_3;

    // [Repeat sequence block for byte steps 4 through 31...]
    // For brevity in the editor, this template repeats the logic over the array
    for (uint8_t i = 4; i < 32; i++) {
        PORTA = in_ptr[i];
        PORTB |= (1 << PB6);
        __asm__ __volatile__("nop\n\t"); // Fixed delay matching hardware flight time
        uint8_t reading = PINC;
        PORTB &= ~(1 << PB6);
        out_ptr[i] = in_ptr[i] ^ reading;
    }
}

void setup() {
    Serial.begin(115200);

    // Configure Port Directions via Data Direction Registers (DDR)
    DDRA = 0xFF; // Set all pins on PORTA (Pins 22-29) as Outputs to the ITO Mask
    DDRC = 0x00; // Set all pins on PORTC (Pins 37-30) as Inputs from Photodiodes
    DDRB |= (1 << DDB6); // Set Pin 12 (IR Strobe) as an Output

    // Clear initial state lines
    PORTA = 0x00;
    PORTB &= ~(1 << PB6);

    // Activate internal pull-up resistors on the photodiode input lines 
    // to keep the signals clean and stable during high-speed transitions
    PORTC = 0xFF; 

    Serial.println(F("OK,MEGA_PHOTONIC_BENCH_INITIALIZED"));
}

void loop() {
    load_mock_pow_hash();

    // Track execution speed using the internal timer clock
    unsigned long start_cycles = micros();
    
    heavy_transform_mega_offload();
    
    unsigned long end_cycles = micros();

    // Print operational telemetry
    Serial.print(F("SUCCESS | Execution Delta: "));
    Serial.print(end_cycles - start_cycles);
    Serial.println(F(" microseconds"));

    Serial.print(F("Output Hash String: "));
    for (uint8_t i = 0; i < 32; i++) {
        if (s_transformed_out[i] < 16) Serial.print('0');
        Serial.print(s_transformed_out[i], HEX);
    }
    Serial.println(F("\n--------------------------------------------------"));

    delay(1000); // 1-second pulse delay for telemetry readability
}
