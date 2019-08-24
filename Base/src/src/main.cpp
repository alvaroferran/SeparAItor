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
uint8_t tilt_up = 0;
uint8_t tilt_down = 50;
uint8_t default_angle = 90;


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
    yaw.write(default_angle);
    tilt.write(tilt_up);
}


int read_string(char * p_string, uint8_t max_len){
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
        if(chars_received == max_len){
            received = true;
        }
        chars_received = 0;
    }
	return received;
}


void go_to_pos(char * p_string){
    uint8_t angles[] = {5, 45, 90, 135};    // glass, organic, PMC, paper
    uint8_t bin_selection = (*p_string - '0');
    static uint8_t prev_yaw_angle = default_angle;
    uint8_t yaw_angle = angles[bin_selection];
    int16_t yaw_difference = yaw_angle - prev_yaw_angle;
    uint8_t wait_angle_yaw = 30;
    uint8_t wait_angle_tilt = 10;
    uint16_t wait_tilt = 2000;
    // Go to new bin 
    if(yaw_difference>0){
        for(uint8_t angle=prev_yaw_angle; angle<yaw_angle; angle++){
            yaw.write(angle);
            dbg(angle);
            delay(wait_angle_yaw);
        }
        yaw.write(yaw_angle);
    }else if (yaw_difference<0){
        for(uint8_t angle=prev_yaw_angle; angle>yaw_angle; angle--){
            yaw.write(angle);
            dbg(angle);
            delay(wait_angle_yaw);
        }
        yaw.write(yaw_angle);
    }
    // Drop object
    for(uint8_t angle=tilt_up; angle<tilt_down; angle+=10){
        tilt.write(angle);
        delay(wait_angle_tilt);
    }
    delay(wait_tilt);
     for(uint8_t angle=tilt_down; angle>tilt_up; angle-=10){
        tilt.write(angle);
        delay(wait_angle_tilt);
    }
    tilt.write(tilt_up);
    delay(wait_tilt);
    prev_yaw_angle = yaw_angle;
}


void loop() {
    static bool received_new = false;
    
    if(Serial.available()){
        received_new = read_string(&in_string[0], MAX_LEN_IN);
    }

	if (received_new == true){
        dbg(in_string);
		go_to_pos(&in_string[0]);
        while(1){
            Serial.println("c");
            Serial.flush();
            if(Serial.available()){
                char c = Serial.read();
                if(c == 'd'){
                    break;
                }
            }
        }
        received_new = false;
	}
}
