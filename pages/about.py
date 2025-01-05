import streamlit as st

def main():
    st.set_page_config(
        page_title="About",
        page_icon="img/usep-logo.png",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("About MAS License Plate App")
    
    # About Section
    st.markdown(
        """
        ## About
        This app uses machine learning models to detect and extract text from license plates.
        
        ### Features:
        - **Image Upload**: Upload an image and extract license plate details.
        - **Real-time Camera**: Detect license plates live via your camera feed.
        - **OCR Integration**: Optical Character Recognition (OCR) to extract text with confidence levels.
        
        ### Contact
        For questions or support, please reach out to [support@example.com](mailto:support@example.com).
        """
    )

    # Team Section
    st.markdown("### Meet the Team")

    # Team Members
    team_members = [
        {
            "name": "Yman Fernandez",
            "email": "yrmfernandez00214@usep.edu.ph",
            "avatar": "https://example.com/alice_avatar.jpg"  # Replace with actual avatar URL or file path
        },
        {
            "name": "PJ Figuracion",
            "email": "pacfiguracion01488@usep.edu.ph",
            "avatar": "img/pj_pic.jpg"  # Replace with actual avatar URL or file path
        },
        {
            "name": "Tophy Linganay",
            "email": "charlie@example.com",
            "avatar": "https://example.com/charlie_avatar.jpg"  # Replace with actual avatar URL or file path
        },
        {
            "name": "Armond Lozano",
            "email": "ahdlozano01497@usep.edu.ph",
            "avatar": "img/armond_pic.jpg"  # Replace with actual avatar URL or file path
        },
        {
            "name": "Edryan Manocay",
            "email": "charlie@example.com",
            "avatar": "img/ed_pic.png"  # Replace with actual avatar URL or file path
        },
        {
            "name": "Norlan Mendoza",
            "email": "nbmendoza00217@usep.edu.ph",
            "avatar": "https://example.com/charlie_avatar.jpg"  # Replace with actual avatar URL or file path
        }
    ]
    
    # Display team members
    for member in team_members: 
        st.markdown(f"**{member['name']}**")
        st.image(member["avatar"], width=300)  # Display avatar image
        st.markdown(f"Email: [{member['email']}]({member['email']})")  # Display email with link

if __name__ == "__main__":
    main()
