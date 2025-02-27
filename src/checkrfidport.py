import serial.tools.list_ports
from flask import jsonify
import os
import time
import re

def check_rfid_available():
    """Endpoint to check RFID reader availability"""
    return jsonify({
        "available": check_rfid_reader(),
        "message": "RFID reader detected" if check_rfid_reader() else "No RFID reader found"
    })

def check_rfid_reader():
    """Check if RFID reader is connected via USB"""
    try:
        # Look for USB devices that might be the RFID reader
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if ("AuthenTec" in port.description or 
                "SYC ID&IC USB Reader" in port.description or 
                "08FF:0009" in port.hwid.upper()):
                return True
                
        # Alternative method: check if hidraw1 exists in /dev
        if os.path.exists("/dev/hidraw1"):
            return True
            
        return False
    except Exception as e:
        print(f"Error checking RFID reader: {e}")
        return False

def read_rfid_card(timeout=15):
    """Read RFID card and return the ID"""
    if not check_rfid_reader():
        print("No RFID reader detected")
        return None
    
    # Open the HID device
    try:
        rfid_port = open("/dev/hidraw1", "rb")
    except Exception as e:
        print(f"Could not open RFID reader: {e}")
        return None
    
    # Read loop
    buffer = ""
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < timeout:
            # Read data from HID device
            byte_data = rfid_port.read(8)
            
            if byte_data and any(b != 0 for b in byte_data):
                # Extract digits from the raw data
                for i in range(len(byte_data)):
                    if 48 <= byte_data[i] <= 57:  # ASCII codes for 0-9
                        buffer += chr(byte_data[i])
                
                # Look for 9-digit card ID pattern
                card_ids = re.findall(r'\d{9}', buffer)
                if card_ids:
                    rfid_port.close()
                    return card_ids[-1]  # Return the most recent card ID
                
                # Keep buffer from growing too large
                if len(buffer) > 100:
                    buffer = buffer[-50:]
            
            time.sleep(0.01)
        
        # Timeout reached
        print("Timeout waiting for card")
        return None
            
    except Exception as e:
        print(f"Error reading RFID card: {e}")
        return None
    finally:
        rfid_port.close()

def test_rfid_reader():
    """Simple test function to read and display card ID"""
    print("RFID Reader Test - Scan a card (Ctrl+C to exit)")
    
    try:
        while True:
            print("Waiting for card...")
            card_id = read_rfid_card()
            if card_id:
                print(f"🔖 Card ID: {card_id}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nTest ended by user")

if __name__ == "__main__":
    test_rfid_reader()