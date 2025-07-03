# Smart Document Knowledge Base
**Student Name:** Petrisa Čanji Vouri
**Date:** 3rd of July 2025

## Features Implemented

### Day 2 Features (Choose 3):
- [ ] Source Attribution
- [ ] Document Manager  
- [ ] Search History
- [ ] Document Statistics
- [ ] Tabbed Interface
- [ ] Advanced Search
- [ ] Export Functionality
- [ ] Enhanced Error Handling

**My chosen features:**
1. Source Attribution - Automatically displays which documents the answers were pulled from by extracting metadata from document chunks, helping users understand and verify the origin of the information.

2. Document Manager  - Provides a user-friendly interface for uploading, viewing, and deleting multiple documents, using st.session_state to persist uploads across interactions. 

3. Search History - Stores all user questions and corresponding AI-generated answers along with their sources. Presented in a collapsible section for easy reference and review.

4. Document Statistics - Analyzes uploaded files to show total word count, average words per document, and file type breakdown, giving users a quick overview of their content library.

5. Tabbed Interface - Organizes the app into three clear tabs (Upload & Convert, Ask Questions, Doc Stats), creating a clean and intuitive user experience that separates functionality.

### Day 3 Styling (Choose 3):
- [ ] Color Themes
- [ ] Loading Animations
- [ ] Layout Improvements
- [ ] Icons & Visual Elements
- [ ] Message Styling
- [ ] Headers & Footers
- [ ] Interactive Elements
- [ ] Data Visualization

**My chosen styling:**
1. Color Themes - Custom color for the background of the app and different colour for buttons.

2. Loading Animations - Custom loading messages with playful text and music-themed animations to improve UX.

3. Icons & Visual Elements - Includes music note animations and emojis to match the soundtrack theme and make the app more engaging.

4. Message Styling - Custom buttons, headers, and text colors using custom CSS for a polished and branded appearance.

## How to Run
1. Install required packages: `pip install streamlit chromadb transformers torch docling`
2. Run the app: `streamlit run Final_app2.py`
3. Upload documents and start asking questions!

## Challenges & Solutions
### Streamlit Implementation Report

#### 1. Maintaining Search History Across Tabs

**Challenge:**  
Keeping search history visible and editable without bloating the interface.

**Solution:**  
To address this, st.session_state was utilized to persist search history across tabs, ensuring that users could access and modify their previous queries seamlessly. Additionally, expander elements were implemented in Tab 2, providing a clean and interactive way to display the search history. This approach maintained a streamlined interface while offering full editability and visibility. 

#### 2. Integrating an external APIs in the app

**Challenge:**  
Integrating and using external APIs (specifically the OpenAI ChatGPT API) in a Streamlit app, including handling API keys securely, updating code for new API versions, and managing document uploads and deletions with Streamlit’s session state.

**Solution:**  
To use the OpenAI (ChatGPT) API in Streamlit, first install the OpenAI Python package. Securely store your API key as an environment variable—never hardcode it. Use the new openai.OpenAI client (for versions ≥1.0.0) and call client.chat.completions.create to interact with ChatGPT. In your app, call a function like ask_openai() to get responses based on your document context, ensuring the API key is set before running the app.

#### 3. Custom Styling in Streamlit

**Challenge:**  
Applying advanced CSS in Streamlit applications without interfering with the platform’s native rendering and interactive elements.

**Solution:**  
Scoped blocks were injected directly via st.markdown, allowing for targeted customizations. Careful management of the z-index property ensured that floating elements, such as modals or tooltips, displayed correctly without overlapping or hiding essential UI components. This method enabled advanced styling while preserving the integrity and responsiveness of the Streamlit interface.

## What I Learned
1. Integrating Multiple Libraries
A major takeaway was learning how to seamlessly combine several powerful libraries—Docling, Langchain, Chroma, and OpenAI—within a single Streamlit application. This integration enabled the creation of a robust pipeline for document processing, semantic search, and AI-powered interactions, all within a unified user interface.

2. Managing State and Document Handling
Another important lesson was mastering session state management in Streamlit. This included handling multiple document uploads efficiently and implementing effective data chunking strategies to optimize semantic search performance. By leveraging session state, the app maintained continuity and responsiveness across user interactions.

3. Enhancing User Experience with Custom Styling
Finally, the project highlighted the value of elevating user experience through custom CSS and engaging animations. These enhancements transformed a purely technical tool into a delightful and user-friendly product, demonstrating that thoughtful design can significantly improve usability and satisfaction.

4. Using ChatGPT API key
I learned how to integrate the OpenAI (ChatGPT) API into a Streamlit app by securely managing the API key using environment variables and avoiding hardcoding sensitive information. I gained experience using the updated openai.OpenAI client for versions 1.0.0 and above, including how to call client.chat.completions.create to interact with the ChatGPT model. Additionally, I understood how to structure API calls within the app, such as using a function like ask_openai() to retrieve responses based on document context, ensuring smooth and secure communication with the OpenAI service.