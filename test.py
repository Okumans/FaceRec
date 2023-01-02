import threading
import time
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_finance

# global variable to hold the prices
prices = []

# generate simulated prices in a background thread
def generate_data():
    global prices
    while True:
        price = simulate_price(100, 1)
        prices.append(price)
        time.sleep(0.1)


# create the candle stick graph
def create_graph():
    global prices
    # convert prices to tuples of (time, open, high, low, close)
    data = [(i, prices[i], prices[i], prices[i], prices[i]) for i in range(len(prices))]

    # create the candle stick graph
    fig, ax = plt.subplots()
    mpl_finance.candlestick_ohlc(ax, data)
    return fig, ax


# update the graph data in real time
def update(num, fig, ax):
    global prices
    ax.cla()
    # convert prices to tuples of (time, open, high, low, close)
    data = [(i, prices[i], prices[i], prices[i], prices[i]) for i in range(len(prices))]
    mpl_finance.candlestick_ohlc(ax, data)


# simulate price using random walk method
def simulate_price(start_price, num_steps):
    global prices
    price = start_price
    for i in range(num_steps):
        price += random.uniform(-1, 1)
    return price


# create and start the background thread
thread = threading.Thread(target=generate_data)
thread.start()

# create the graph and animate it
fig, ax = create_graph()
ani = FuncAnimation(fig, update, fargs=(fig, ax), interval=100)
plt.plot()
