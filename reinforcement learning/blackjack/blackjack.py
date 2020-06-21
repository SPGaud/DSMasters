import numpy as np
import copy


class Blackjack:
    def __init__(self, verbose=True):
        # Print game progress or not.
        self.verbose = verbose

        # Card value of 1 is an ace
        self.card_values = np.arange(1, 11)
        self.num_different_cards = len(self.card_values)
        # Calculate probabilities to draw a certain card. Face cards have value of 10
        self.card_frequencies = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 16])
        self.card_probabilities = self.card_frequencies / np.sum(self.card_frequencies)

        self.player_cards = []
        self.player_sum = 0
        self.player_has_usable_ace = False

        self.dealer_cards = []
        self.dealers_showing_card = 0
        self.dealer_sum = 0

        self.state = np.array([int(self.player_sum), self.dealers_showing_card, self.player_has_usable_ace])
        self.active = False

    def reset_game(self):
        """
        Reset all game variables to their initial states and draw cards for both player and dealer.
        """
        if self.verbose:
            print("The game is reset.")
        self.active = True
        self.player_cards = []
        self.player_sum = 0
        self.player_has_usable_ace = False
        while self.player_sum < 12:
            new_card = self.draw_cards(size=1)
            if new_card == 1 and self.player_sum < 11:
                new_card = 11
                self.player_has_usable_ace = True
            self.player_cards = np.append(self.player_cards, new_card)
            self.player_sum += new_card

        self.dealer_cards = self.draw_cards(size=2)
        self.dealer_sum = np.sum(self.dealer_cards)
        self.dealers_showing_card = self.dealer_cards[1]

        self.state = np.array([int(self.player_sum), self.dealers_showing_card, self.player_has_usable_ace])
        if self.verbose:
            print("Player's cards:", [int(x) for x in self.player_cards]) # , "; player sum:", self.player_sum[0]
            print("Dealer's showing card:", [self.dealer_cards[1]])

    def draw_cards(self, size=1):
        return np.random.choice(a=self.card_values, size=size, p=self.card_probabilities)

    def make_step(self, action="reset"):
        """
        Compute the response of the environment given an action.

        The function is used during the game and takes the player's 'action' ("hit" or "stick") as input.
        It is also used to start / reset a game and checks whether the episode ends

        Returns a new state and a reward signal.
        """
        reward = 0
        if not self.active and action != "reset":
            print("You specified an action, but no active game is being played. Please reset the game using action='reset'.")
        elif self.active and action == "reset":
            print("WARNING! You specified action='reset' although the last game has not ended, yet.")
        elif action == "reset":
            self.reset_game()
            if self.player_sum == 21:
                self.active = False
                if self.verbose:
                    print("Player has Blackjack!")
                    print("The dealer's cards are:", [int(x) for x in self.dealer_cards])
                if np.sum(self.dealer_cards == 1) == 1 and np.sum(self.dealer_cards == 10) == 1:
                    if self.verbose:
                        print("Dealer has Blackjack, too!")
                        print("DRAW!")
                else:
                    if self.verbose:
                        print("PLAYER WINS!")
                    reward = 1
        elif action == "hit":
            new_card = self.draw_cards(size=1)
            if self.verbose:
                print("Player draws card:", new_card)
            self.player_sum += new_card
            self.player_cards = np.append(self.player_cards, new_card)
            if self.player_sum > 21:
                if self.player_has_usable_ace:
                    if self.verbose:
                        print("Player converts a usable ace (11) into 1.")
                    self.player_cards[self.player_cards == 11] = 1
                    self.player_has_usable_ace = False
                    self.player_sum -= 10
            if self.player_sum > 21:
                # Player goes bust
                if self.verbose:
                    print("Player goes BUST!")
                self.active = False
                reward = -1
            else:
                self.state = np.array([int(self.player_sum), self.dealers_showing_card, self.player_has_usable_ace])
                if self.verbose:
                    print("New sum of player's cards:", self.player_sum)
                reward = 0
        elif action == "stick":
            self.active = False
            if self.verbose:
                print("The dealer's cards are:", [int(x) for x in self.dealer_cards])
                print("The dealer has", np.sum(self.dealer_cards), "points.")
            # Let dealer play
            if np.any(self.dealer_cards == 1) and 17 <= self.dealer_sum + 10 <= 21:
                if self.verbose:
                    print("Dealer converts 1 into 11")
                self.dealer_sum += 10
                self.dealer_cards[self.dealer_cards == 1] = 11
            while self.dealer_sum < 17:
                new_card = self.draw_cards(size=1)
                if self.verbose:
                    print("Dealer draws card:", new_card)
                self.dealer_cards = np.append(self.dealer_cards, new_card)
                self.dealer_sum += new_card
                if np.any(self.dealer_cards == 1) and 17 <= self.dealer_sum + 10 <= 21:
                    if self.verbose:
                        print("Dealer converts 1 into 11")
                    self.dealer_sum += 10
                    self.dealer_cards[np.where(self.dealer_cards == 1)[0]] = 11
                if self.verbose:
                    print("New dealer sum", self.dealer_sum)

            # Check whether dealer goes bust.
            dealer_difference = 21 - self.dealer_sum
            if dealer_difference < 0:
                if self.verbose:
                    print("Dealer goes BUST!")
                reward = 1
            else:
                # Compare player to dealer
                player_difference = 21 - self.player_sum
                if player_difference == dealer_difference:
                    if self.verbose:
                        print("DRAW!")
                    reward = 0
                elif player_difference < dealer_difference:
                    if self.verbose:
                        print("PLAYER WINS!")
                    reward = 1
                else:
                    if self.verbose:
                        print("DEALER WINS!")
                    reward = -1
        else:
            raise ValueError("'action' has to be either 'stick' or 'hit' when gameover is False.")

        if not self.active:
            self.state = np.array([-1, -1, -1])

        return copy.deepcopy(self.state), reward


