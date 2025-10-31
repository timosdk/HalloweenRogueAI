import OPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

pin = 12
GPIO.setup(pin, GPIO.OUT)

try:
 while True:
  print("High")
  GPIO.output(pin, GPIO.HIGH)
  time.sleep(1)
  print("Low")
  GPIO.output(pin, GPIO.LOW)
  time.sleep(1)

except KeyboardInterrupt:
 print("Exit")
 exit()
