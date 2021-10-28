board = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
win_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8],[0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
def show():
    print(board[0], "|", board[1], "|", board[2])
    print("--|---|--")
    print(board[3], "|", board[4], "|", board[5])
    print("--|---|--")
    print(board[6], "|", board[7], "|", board[8])
    
def player1():
    n = choose_position()
    if board[n] == "X" or board[n] == "0":
        print("This place is filled")
        player1()
    else:
        board[n] = "0"

def player2():
    n = choose_position()
    if board[n] == "X" or board[n] == "0":
        print("This place is filled. Try again.")
        player2()
    else:
        board[n] = "X"

def choose_position():
    while True:
        a = int(input())
        a -= 1
        try: 
            if a in range(0, 9):
                return(a)
            else:
                print("This position is not on board. Try again:")
                continue
        except ValueError:
            print("That's not a number. Try again:")
            continue
            
def check_board():
    for i in win_combinations:
        if board[i[0]] == "0" and board[i[1]] == "0" and board[i[2]] == "0":
            print("Player1 wins!!")
            return(1)
        elif board[i[0]] == "X" and board[i[1]] == "X" and board[i[2]] == "X":
            print("Player2 wins!!")
            return(1)
            
    t = 0
    for i in range(9):
        if board[i] == "0" or board[i] == "X":
            t += 1
        if t == 9:
            print("Game draw!!")
            return(-1)
            
p = "y"
q = 0
while p == "y":
    board = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    q = check_board()
    while q != -1:
        show()
        q = check_board()
        if q == 1 or q == -1:
            break
        print("Player1 choose where to enter: ")
        player1()
        show()
        q = check_board()
        if q == 1 or q == -1:
            break
        print("Player2 choose where to enter: ")
        player2()
    p = input("Enter 'y' to play again or enter any other key to exit: ")
        
        
    
            

            

            