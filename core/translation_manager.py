from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


class TranslationManager:
    def __init__(self, google_api_key, model="gemini-2.0-flash"):
        # Initialize Gemini LLM via Langchain
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key)

        # Define the translation prompt template
        self.translation_prompt = PromptTemplate(
            input_variables=["text", "target_language"],
            template="Translate the following text to {target_language}: {text}"
        )

    def translate(self, text, target_language):
        if target_language == "None" or target_language == "English":
            return text
        
        # Prepare the translation prompt for Gemini
        prompt = self.translation_prompt.format(text=text, target_language=target_language)
        
        # Request translation from Google Gemini LLM
        try:
            response = self.llm(prompt)
            translated_text = response["result"]
            return translated_text
        except Exception as e:
            return f"Error during translation: {str(e)}"
