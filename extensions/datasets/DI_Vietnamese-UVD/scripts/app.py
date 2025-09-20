import streamlit as st


def main():
    st.title("DI_Vietnamese-UVD")

    message = st.text_area("Enter Your Text", "Type Here")
    if st.button("Analyze"):
        st.success(message.title())


if __name__ == '__main__':
    main()
