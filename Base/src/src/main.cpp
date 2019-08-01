#include <Arduino.h>
#include "Servo.h"
#include <string.h>

// #define DEBUG  // Uncomment to get debug messages

#ifdef DEBUG
    #define dbg(x) Serial.println(x)
    #define dbgc(x) Serial.print(x)
#else
    #define dbg(x)
    #define dbgc(x)
#endif

# define MAX_LEN_IN 2  // 0\n


char in_string[MAX_LEN_IN];
uint8_t pin_yaw = 8;
uint8_t pin_tilt = 7;
Servo yaw, tilt;
uint8_t tilt_min = 0;
uint8_t tilt_max = 50;
uint8_t angles[] = {90, 70, 80, 100, 110}; // Default, PMD, organic, glass, paper


void setup() {
    memset (&in_string[0], '0', MAX_LEN_IN*sizeof(uint8_t));
    Serial.begin(19200);
    // Simple handshake to verify connection
    while(1){
        Serial.println("a");
        char c = Serial.read();
        if(c == 'b'){
            break;
        }
    }
    yaw.attach(pin_yaw);
    tilt.attach(pin_tilt);
    yaw.write(angles[0]);
    tilt.write(tilt_min);
}


int read_string(char * p_string){
	char in_char;
    char end_char = '\n';
    bool received = false;
    static uint8_t chars_received = 0;

    in_char = (char) Serial.read();
    if(in_char != end_char){
        p_string[chars_received++] = in_char;
    }
    else{
        p_string[chars_received++] = '\0';
        if(chars_received == MAX_LEN_IN){
            received = true;
        }
        chars_received = 0;
    }
	return received;
}


void go_to_pos(char * p_string){
    uint8_t bin_selection = (*p_string - '0') +1;
    uint16_t wait_ms = 1000;
    yaw.write(angles[bin_selection]);
    delay(wait_ms);
    tilt.write(tilt_max);
    delay(wait_ms);
    tilt.write(tilt_min);
    yaw.write(angles[0]);
    delay(wait_ms);
}


void loop() {
    static bool received_new = false;
    
    if(Serial.available()){
        received_new = read_string(&in_string[0]);
    }

	if (received_new == true){
        dbg(in_string);
		go_to_pos(&in_string[0]);
        while(1){
            Serial.println("a");
            char c = Serial.read();
            if(c == 'b'){
                break;
            }
        }
        received_new = false;
	}
}
