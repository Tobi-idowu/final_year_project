import time

def print_every_ten_seconds():
    count = 0
    try:
        while True:
            print(f"The count is {count}")
            count += 1
            time.sleep(10)  # Wait for 10 seconds
    except KeyboardInterrupt:
        print("Program stopped by user.")

# Run the function
print_every_ten_seconds()
