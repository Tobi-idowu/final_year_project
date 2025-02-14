import time

def print_every_ten_seconds():
    count = 0
    try:
        while True:
            print(f"The count is {count}")
            count += 1
            time.sleep(10)  #wait for 10 seconds
    except KeyboardInterrupt:
        print("Program stopped by user.")

#run the function
print_every_ten_seconds()
