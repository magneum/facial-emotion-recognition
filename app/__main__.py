# by M.A.G.N.E.U.M
# This code builds a facial emotion recognition model using tensorflow keras and fer2013

import os
import colorama
from routes import train
from routes import img_pred
from routes import vid_pred
from routes import live_pred


def main():
    os.system("clear")
    colorama.init()
    while True:
        choice = input(
            colorama.Fore.YELLOW
            + "What would you like to do?\n"
            + colorama.Fore.RESET
            + "1. Image prediction\n"
            + colorama.Fore.RESET
            + "2. Video prediction\n"
            + colorama.Fore.RESET
            + "3. Live prediction\n"
            + colorama.Fore.RESET
            + "4. Train model\n"
            + colorama.Fore.RESET
            + "Enter your choice: "
        )

        if choice == "1":
            img_pred.main()
        elif choice == "2":
            vid_pred.main()
        elif choice == "3":
            live_pred.main()
        elif choice == "4":
            train.main()
        else:
            print(colorama.Fore.RED + "Invalid choice." + colorama.Fore.RESET)
        continue_choice = input("Would you like to continue? (y/n): ")
        if continue_choice == "n":
            break


if __name__ == "__main__":
    main()
