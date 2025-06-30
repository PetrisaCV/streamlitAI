# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers
# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Custom CSS for button styling 
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #A32CC4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 🎵 Your app starts here

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        "Soundtrack 1: The Music of Hans Zimmer: Hans Zimmer is one of the most prolific and influential film composers of the modern era. Born in Frankfurt, Germany, in 1957, Zimmer’s career took off in the late 1980s with his Oscar-nominated score for Rain Man (1988). His signature style combines orchestral and electronic elements, often emphasizing emotion through layered textures and rhythmic motifs. Zimmer’s score for The Lion King (1994) earned him an Academy Award, and his collaboration with director Christopher Nolan on Inception (2010) and Interstellar (2014) pushed the boundaries of cinematic sound. The use of a Shepard tone illusion in Dunkirk (2017), for example, creates a sense of perpetual tension. Zimmer’s work on Gladiator, Pirates of the Caribbean, and Dune showcases his range, from intimate melodies to thunderous battle themes. Zimmer has scored over 150 films and continues to inspire a generation of composers. He often collaborates with other musicians, such as Lisa Gerrard and Johnny Marr, making his work rich in tone and diversity. His ability to elevate visuals into emotional experiences makes him a cornerstone of modern soundtrack composition.",

        "Soundtrack 2: The Role of Music in Storytelling: Soundtrack music plays a crucial role in storytelling by shaping the viewer’s emotional response and enhancing narrative depth. Film music can signal mood, indicate character development, or foreshadow events. For example, the ominous two-note motif in Jaws (1975) by John Williams is a textbook case of how music builds suspense. Leitmotifs—short, recurring musical phrases associated with characters or ideas—are widely used in storytelling. In Star Wars, each major character has a theme: Darth Vader’s “Imperial March” evokes dread, while Leia’s theme signals grace and hope. These motifs act as sonic signposts, helping audiences navigate the story emotionally. Music can also bridge scenes, mark transitions, or contrast with visuals to create irony. Quentin Tarantino often uses upbeat tracks during violent scenes, enhancing their surreal intensity. In animation, like Pixar’s Up, Michael Giacchino’s score tells a love story almost without words, using melody to span decades of emotion. Ultimately, music turns passive viewing into an immersive experience. A well-composed score can make a scene unforgettable—sometimes even more than the dialogue.",

        "Soundtrack 3: The Evolution of Soundtracks Through Time: The soundtrack has evolved significantly since the silent film era. In the 1910s and 1920s, live pianists or small orchestras provided improvised accompaniment in theaters. With the advent of synchronized sound in the late 1920s—thanks to The Jazz Singer (1927)—composers began writing music specifically for films. The Golden Age of Hollywood (1930s–1950s) saw the rise of lush, orchestral scores by composers like Max Steiner (Gone with the Wind) and Bernard Herrmann (Psycho). In the 1970s, John Williams revived the grand symphonic style with Star Wars, which remains a benchmark in film scoring. The 1980s and 1990s brought greater experimentation, with electronic elements introduced by composers like Vangelis (Blade Runner) and hybrid soundtracks by Hans Zimmer. In the 2000s and 2010s, composers embraced digital production, sampling, and non-traditional instruments to reflect diverse storytelling styles. Streaming platforms and video games have also influenced soundtracks. Series like Stranger Things (with its synth-heavy score) show how music can define an era. Today’s soundtracks blend genres, cultures, and technologies—demonstrating that film music is not only evolving but thriving as an art form.",

        "Soundtrack 4: Original Scores vs. Curated Soundtracks: Soundtracks come in two primary forms: original scores and curated soundtracks. An original score is music composed specifically for a film, tailored to match its tone, pacing, and emotional beats. A curated soundtrack uses pre-existing songs—often popular music—selected to enhance specific scenes. Original scores offer the advantage of narrative cohesion. Think of Howard Shore’s The Lord of the Rings trilogy, where themes evolve with the story and characters. These scores are often symphonic and thematic, designed to support storytelling on a deep, emotional level. Curated soundtracks, on the other hand, can bring cultural context or nostalgic power. In Guardians of the Galaxy, the Awesome Mix tapes play a central narrative role, connecting the protagonist to his past while energizing the film’s pace. Quentin Tarantino is also known for using eclectic songs to add irony or emotional punch. Both approaches have their merits. Some films combine them—Black Panther (2018) featured both Ludwig Göransson’s original score and a curated album by Kendrick Lamar. Whether composed or compiled, a great soundtrack serves the same purpose: to elevate the film experience through music.",

        "Soundtrack 5: Underrated Soundtracks That Deserve More Attention: While blockbuster films like Star Wars or Inception dominate the spotlight, many excellent soundtracks fly under the radar. These scores may come from indie films, international cinema, or overlooked genres, yet they offer rich musical experiences worth exploring. One such gem is The Assassination of Jesse James by the Coward Robert Ford (2007), composed by Nick Cave and Warren Ellis. Its sparse, haunting piano melodies perfectly match the film’s slow-burn aesthetic. Another is Moonlight (2016), where Nicholas Britell’s score blends classical elements with chopped-and-screwed hip-hop textures, mirroring the protagonist’s identity struggles. Animated films also produce underrated work. Joe Hisaishi’s collaborations with Studio Ghibli, like Princess Mononoke or Spirited Away, feature sweeping, emotional scores that rival any Hollywood production. Similarly, Her (2013), scored by Arcade Fire, uses ambient textures to explore human-technology relationships. Exploring lesser-known soundtracks broadens our appreciation of film music and highlights composers who innovate outside the mainstream. Whether you're a casual listener or soundtrack enthusiast, these scores offer something fresh and emotionally powerful."
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["soundtrack1", "soundtrack2", "soundtrack3", "soundtrack4", "soundtrack5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
# 🎬 Title
st.markdown("## 🎵 Welcome to the **Soundtrack Explorer**")
st.markdown("*Your backstage pass to film music magic!*")

# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
# 💬 Intro
st.write("Welcome to my curated hub for all things soundtrack-related! Dive into the world of film music and learn about composers, trends, and hidden gems.")
# This text appears below the title and gives context to the app

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
# 📝 Input label
question = st.text_input("🎬 What soundtrack mystery would you like to uncover today?")
# Placeholder text appears inside the box when empty

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
# 🔍 Button
if st.button("🔍 Reveal the Music Magic", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Thinking and humming Hakuna matata..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")


# STREAMLIT BUILDING BLOCK 6: SPINNER

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
# ℹ️ Help section
with st.expander("ℹ️ What can I ask about?"):
    st.markdown("""
    🎼 This app is all about **soundtracks in cinema**.
    
    You can ask about:
    - 🎹 Famous composers like *Hans Zimmer*
    - 🎭 How music tells a story in films
    - ⏳ Soundtrack history and trends
    - 📀 The difference between curated playlists vs. original scores
    - 🕵️‍♀️ Underrated scores you might have missed
    
    💡 *Tip:* Ask about a specific theme to unlock custom insights!
    """)
    st.info("💡 Tip: Start with a keyword like *Zimmer*, *story*, or *underrated* to explore specific topics.")

# TO RUN: Save as app.py, then type: streamlit run app.py

"""
STREAMLIT BUILDING BLOCKS SUMMARY:
================================

1. st.title(text) 
   - Creates the main heading of your app
   - Appears as large, bold text at the top

2. st.write(text)
   - Displays text, data, or markdown content
   - Most versatile output function in Streamlit
   - Can display simple text, formatted text, or data

3. st.text_input(label, placeholder="hint")
   - Creates a text box where users can type
   - Returns whatever the user types
   - Label appears above the box

4. st.button(text, type="primary")
   - Creates a clickable button
   - Returns True when clicked, False otherwise
   - Use in 'if' statements to trigger actions
   - type="primary" makes it blue and prominent

5. st.spinner(text)
   - Shows a spinning animation with custom text
   - Use with 'with' statement for code that takes time
   - Automatically disappears when code finishes

6. st.expander(title)
   - Creates a collapsible section
   - Users can click to expand/collapse content
   - Great for help text or optional information
   - Use with 'with' statement for content inside

HOW THE APP FLOW WORKS:
======================

1. User opens browser → Streamlit loads the app
2. setup_documents() runs → Creates document database
3. st.title() and st.write() → Display app header
4. st.text_input() → Shows input box for questions  
5. st.button() → Shows the "Get Answer" button
6. User types question and clicks button:
   - if statement triggers
   - st.spinner() shows loading animation
   - get_answer() function runs in background
   - st.write() displays the result
7. st.expander() → Shows help section at bottom

WHAT HAPPENS WHEN USER INTERACTS:
=================================

- Type in text box → question variable updates automatically
- Click button → if st.button() becomes True
- Spinner shows → get_answer() function runs
- Answer appears → st.write() displays the result
- Click expander → help section shows/hides

This creates a simple but complete web application!
"""
