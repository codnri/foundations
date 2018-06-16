import itertools
from random import shuffle

class Participant:
  def __init__(self):
    self.hand = []
    self.total = 0

  def calculate_total(self):
    cards = self.hand
    total = 0
    for(suit,value) in cards:
      if value == 'A':
        total += 11
      elif value in ['J','Q','K']:
        total += 10
      else:
        total += int(value)

    aces = sum(card.count('A') for card in cards )

    while total > 21 and aces:
      total -=10
      aces -=1

    self.total = total
    pass
  
  def hit(self,card):
    self.hand.append(card)
    self.calculate_total()
  
  def showAllCards(self):
    for card in self.hand:
      print(card)
    print("")



print("Start BlackJack")
suits=['H','D','S','C']
cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

deck= list(itertools.product(suits,cards))
shuffle(deck)

# print(deck)


player = Participant()
dealer = Participant()

player.hit(deck.pop())
dealer.hit(deck.pop())
player.hit(deck.pop())
dealer.hit(deck.pop())

print("Dealer has: Hidden and {0}".format(dealer.hand[1]))
print("You have {0} and {1}, for a total of {2}".format(player.hand[0], player.hand[1],player.total))

# Check if there is a winner
if player.total == 21 and dealer.total == 21:
    print("It's a push. House wins. Sorry.")
    quit()
elif player.total == 21:
    print("Congratulations, you hit Blackjack! You win!")
    quit()
elif dealer.total == 21:
    print("Sorry, dealer hit Blackjack. You lose.")
    quit()
else:
    pass #explicit the ending


# Player Turn
while player.total < 21:
    hit_or_stay = input("What would you like to do? 1) Hit 2) Stay    ")

    if hit_or_stay not in ['1', '2']:
        print("Error: You must enter 1 or 2.")
        continue
    elif hit_or_stay == "2":
        print("You chose to stay.")
        break
    else:
        new_card = deck.pop()
        print("Dealing card to player: {0}".format(new_card))
        player.hit(new_card)
        # player.calculate_total()
        print("Your total is now: {0}".format(player.total))

        if player.total == 21:
            print("Congratulations, you hit Blackjack! You win!")
            quit()
        elif player.total > 21:
            print("Sorry, you busted!")
            quit()
        else:
            continue

# Dealer Turn
while dealer.total < 17:
    new_card = deck.pop()
    print("Dealing card to dealer: {0}".format(new_card))
    dealer.hit(new_card)
    # dealer.calculate_total()
    print("Dealer's total is now: {0}".format(dealer.total))

    if dealer.total == 21:
        print("Sorry, dealer hit Blackjack. You lose.")
        quit()
    elif dealer.total > 21:
        print("Congratulations, dealer busted! You win!")
        quit()
    else:
        continue


# Compare Hands
print("Dealer's cards: ")
dealer.showAllCards()

print("Your cards: ")
player.showAllCards()

if dealer.total > player.total:
    print("Sorry, dealer wins.")
elif dealer.total < player.total:
    print("Congratulations, you win!")
else:
    print("It's a tie")

quit()

