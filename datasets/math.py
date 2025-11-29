"""Math word problem dataset."""

import dspy
from typing import List, Tuple


TRAIN_DATA = [
    ("Sarah has 15 apples and buys 23 more. How many apples does she have?", "38"),
    ("There are 100 students and 35 went home early. How many students remain?", "65"),
    ("A box contains 12 chocolates. How many chocolates are in 8 boxes?", "96"),
    ("240 cookies are divided equally among 6 children. How many does each child get?", "40"),
    ("Tom bought 5 books at $12 each and a pen for $3. How much did he spend?", "63"),
    ("Jane had $200, spent $45, then doubled her remaining money. How much does she have now?", "310"),
    ("What is 20 plus 30, all divided by 5?", "10"),
    ("A class has 25 students. Each student needs 4 pencils and 2 erasers. How many items total?", "150"),
    ("If 100 candies are shared among 8 kids equally, how many does each get?", "12.5"),
    ("A store had 500 items, sold 120 on Monday and twice that amount on Tuesday. How many remain?", "140"),
]

DEV_DATA = [
    ("A garden has 7 rows with 9 plants each. How many plants total?", "63"),
    ("180 apples divided into 6 baskets, then 5 more apples added to each basket. How many per basket?", "35"),
    ("Start with 1000, subtract 250, then subtract 175. What remains?", "575"),
    ("What is 15 times 8?", "120"),
    ("Calculate: 50 plus 25, then multiply by 3, then subtract 20.", "205"),
]


def get_data():
    """Get math word problem train and dev datasets."""
    train = []
    for problem, answer in TRAIN_DATA:
        ex = dspy.Example(problem=problem, answer=answer)
        train.append(ex.with_inputs("problem"))

    dev = []
    for problem, answer in DEV_DATA:
        ex = dspy.Example(problem=problem, answer=answer)
        dev.append(ex.with_inputs("problem"))

    return train, dev
