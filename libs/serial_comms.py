import serial


def connect_serial(port, baudrate):
    ser = serial.Serial(port, baudrate)
    # Simple handshake to verify connection
    while True:
        in_data = ser.read_until().decode()
        if in_data[:-2] == "a":
            ser.write("b".encode())
            break
    return ser


def send_data(ser, sorted_class):
    data = str(sorted_class) + "\n"
    ser.write(data.encode())
    ser.flush()
