import streamlit as st
from stock import StockData


def main():
    st.set_page_config(page_title="Stock market GBM", layout="wide")

    st.title("Geometric Brownian motion Simulation for given stock")
    st.subheader("By Utkarsh Gaikwad")

    symbol = st.text_input("Please enter a ticker :")

    button = st.button("submit", type="primary")
    if button:
        with st.spinner():
            stock = StockData(ticker=symbol)
            fig1 = stock.plot_data()
            st.pyplot(fig1)

            fig2 = stock.plot_simulations()
            st.pyplot(fig2)


if __name__ == "__main__":
    main()
