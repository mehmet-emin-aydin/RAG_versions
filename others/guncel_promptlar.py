"""String and custom templates for LLM prompts"""
from typing import List, Any

from langchain.prompts import StringPromptTemplate, PromptTemplate
from pydantic.v1 import BaseModel

from app.config import settings
from app.models.database_models import (
    ChatMessage as DbChatMessage,
    ChatMessageRequestType as DbChatMessageType,
)

SYSTEM_PROMPT = (
    "welcome to Code Foundry 👋! I'm your trusty assistant, here to answer your questions, "
    "provide explanations, and even generate code. "
    "So, go ahead and start chatting with me below!\n\n"
    "You can also do some other really cool things! For example, highlight a piece of code and:\n\n"
    "- Explain Code Snippet: `\u2303 \u2318 E` (Mac) or `Shift+Alt+E` (Windows)\n"
    "- Generate Documentation: `\u2303 \u2318 S` or `Shift+Alt+S`\n"
    "- Generate Unit Tests: `\u2303 \u2318 T` or `Shift+Alt+T`\n"
    + (
        "- Refactor Code: `\u2303 \u2318 R` or `Shift+Alt+R`\n\n"
        if not settings.disable_code_refactor
        else "\n"
    )
    + "Just like you 😊, I am not perfect, so surprises and mistakes are possible. "
    "Please verify any generated code or suggestions, and modify as needed. Let's get started!"
)

PROMPT_LIBRARY = [
    "Write a Python script that connects to a MongoDB database, retrieves documents from a "
    "'users' collection, and prints usernames and their corresponding email addresses.",
    "Create a JavaScript function that fetches data from a REST API endpoint, dynamically "
    "populates an HTML table with the results, and updates the UI asynchronously.",
    "Develop a GoLang HTTP server that serves a simple REST API for managing a collection of "
    "books. Implement CRUD operations (Create, Read, Update, Delete) for book entities, "
    "storing data in an in-memory data structure.",
    "Create a front-end code snippet for a registration form with interactive validation. "
    "Implement real-time feedback for users, validating fields such as mail format, password "
    "strength, and ensuring required fields are filled. Display error messages and highlight the "
    "corresponding fields dynamically.",
    "Develop JUnit test cases for a Java class that represents a simple calculator with methods "
    "for addition, subtraction, multiplication, and division.",
    "Write Golang test functions for a package that handles image processing. Include test cases "
    "for image resizing and format conversion.",
    "Write JUnit test cases for a Java class that implements a stack data structure. Include "
    "tests for push, pop, and peek operations, as well as handling edge cases such as an empty "
    "stack.",
    "Add comments to a C++ class representing a file handling utility. Include comments for "
    "methods handling file reading, writing, and error handling.",
    "Enhance a JavaScript module for form validation by adding inline comments to explain the "
    "purpose and usage of each validation function.",
    "Document a Python function that calculates the Fibonacci sequence up to a given limit. "
    "Explain the algorithm used, expected input, output, and any important considerations for "
    "users.",
]

QUESTION_ANSWER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "{context}{history}Question:\n{prompt}\n\nAnswer:\n"
)

CONTEXT_PROMPT_TEMPLATE = PromptTemplate.from_template("Context:\n{context}\n\n")


class UseCasePromptTemplate(StringPromptTemplate, BaseModel):
    """Custom template with complex logic to augment prompt based on use cases"""

    input_variables: List[str] = ["prompt", "in_chat", "use_case_type", "metadata"]

    # Simple prompts
    ORIGINAL_PROMPT_TEMPLATE = PromptTemplate(
        template="{prompt}", input_variables=["prompt"]
    )
    REFACTOR_PROMPT_TEMPLATE = PromptTemplate(
        template="Refactor the following code:\n{prompt}", input_variables=["prompt"]
    )
    TEST_PROMPT_TEMPLATE = PromptTemplate(
        template="Generate unit tests for the following code:\n{prompt}",
        input_variables=["prompt"],
    )
    DOCUMENTATION_PROMPT_TEMPLATE = PromptTemplate(
        template="Describe the functionality and logic behind this code snippet:\n{prompt}",
        input_variables=["prompt"],
    )
    EXPLAIN_PROMPT_TEMPLATE = PromptTemplate(
        template="Explain the following code:\n{prompt}", input_variables=["prompt"]
    )

    # Complex prompts
    COMPLETION_PROMPT_TEMPLATE = PromptTemplate(
        template="{language}Complete the code snippet:\n{prompt}",
        input_variables=["prompt"],
    )
    LANGUAGE_PROMPT_TEMPLATE = PromptTemplate(
        template="{language}{prompt}", input_variables=["prompt"]
    )
    DOCUMENTATION_DETAILED_PROMPT_TEMPLATE = PromptTemplate(
        template="Generate documentation with markdown formatting for the following code. "
        "Documentation and inline Comments should cover function purposes, attributes, "
        "return types, and potential exceptions and edge cases:\n{prompt}",
        input_variables=["prompt"],
    )
    TEST_DETAILED_PROMPT_TEMPLATE = PromptTemplate(
        template="Unit tests with markdown formatting for the following code. Add a comment "
        "describing what the test does, then the test code. Do not include TODO "
        "comments. Every method in the code should have at least one test.{"
        "testing_framework}:\n{prompt}",
        input_variables=["prompt"],
    )
    CHAT_CODE_FORMAT_PROMPT_TEMPLATE = PromptTemplate(
        template="{language}{prompt}. Append markdown formatting for code.{testing_framework}",
        input_variables=["prompt"],
    )

    def format(self, **kwargs: Any) -> (str, str):
        prompt: str = kwargs["prompt"]
        in_chat: bool = kwargs["in_chat"]
        use_case_type: DbChatMessageType = kwargs["use_case_type"]
        metadata: dict | None = kwargs["metadata"]
        return self.augment_prompt(prompt, use_case_type, in_chat, metadata)

    def augment_prompt(
        self,
        prompt: str,
        use_case_type: DbChatMessageType,
        in_chat: bool,
        metadata: dict | None,
    ) -> (str, str):
        """Augment the prompt with additional instructions"""
        testing_framework_str = (
            f" The testing framework is {metadata.get('test_framework')}."
            if metadata is not None and metadata.get("test_framework")
            else ""
        )
        language = (
            f"{metadata.get('language')} "
            if metadata is not None and metadata.get("language")
            else ""
        )

        display_prompt = self.get_display_prompt_template(
            use_case_type, in_chat
        ).format(prompt=prompt)
        actual_prompt_template = self.get_actual_prompt_template(use_case_type, in_chat)

        params = {"prompt": prompt}
        if "testing_framework" in actual_prompt_template.input_variables:
            params["testing_framework"] = testing_framework_str
        if "language" in actual_prompt_template.input_variables:
            params["language"] = language
        actual_prompt = actual_prompt_template.format(**params)

        return actual_prompt, display_prompt

    def get_display_prompt_template(
        self, use_case_type: DbChatMessageType, in_chat: bool
    ) -> PromptTemplate:
        """Get simple prompt template based on use case"""
        if in_chat:
            template = self.ORIGINAL_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.TEST_GENERATION:
            template = self.TEST_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.EXPLANATION:
            template = self.EXPLAIN_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.DOCUMENTATION:
            template = self.DOCUMENTATION_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.CODE_REFACTOR:
            template = self.REFACTOR_PROMPT_TEMPLATE
        else:
            template = self.ORIGINAL_PROMPT_TEMPLATE

        return template

    def get_actual_prompt_template(
        self, use_case_type: DbChatMessageType, in_chat: bool
    ) -> PromptTemplate:
        """Get actual prompt template to send to LLM based on use case"""
        if in_chat:
            if use_case_type in [
                DbChatMessageType.TEST_GENERATION,
                DbChatMessageType.DOCUMENTATION,
                DbChatMessageType.CODE_REFACTOR,
                DbChatMessageType.CODE_TRANSLATE,
            ]:
                template = self.CHAT_CODE_FORMAT_PROMPT_TEMPLATE
            else:
                template = self.ORIGINAL_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.TEST_GENERATION:
            template = self.TEST_DETAILED_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.EXPLANATION:
            template = self.EXPLAIN_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.DOCUMENTATION:
            template = self.DOCUMENTATION_DETAILED_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.CODE_REFACTOR:
            template = self.REFACTOR_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.INLINE_COMPLETION:
            template = self.COMPLETION_PROMPT_TEMPLATE
        elif use_case_type == DbChatMessageType.COMMENT_TO_CODE:
            template = self.LANGUAGE_PROMPT_TEMPLATE
        else:
            template = self.ORIGINAL_PROMPT_TEMPLATE
        return template

    @property
    def _prompt_type(self):
        return "use-case-prompt"


class ChatMessagesPromptTemplate(StringPromptTemplate, BaseModel):
    """Custom template to convert a list of chat messages into string"""

    input_variables: List[str] = ["chat_messages"]

    def format(self, **kwargs: Any) -> str:
        chat_messages = kwargs["chat_messages"]
        return self.convert_messages_to_string(chat_messages)

    @property
    def _prompt_type(self):
        return "chat-messages-prompt"

    @staticmethod
    def convert_messages_to_string(chat_messages: list[DbChatMessage]) -> str:
        """Convert messages array to the following formatted string:
        Question:
        xxx
        Answer:
        """
        if len(chat_messages) == 0:
            prompt = ""
        else:
            prompt = (
                "\n\n".join(
                    [
                        f"{'Question' if msg.role.value.lower() == 'user' else 'Answer'}: \n"
                        f"{msg.actual_message_payload}"
                        for msg in chat_messages
                    ]
                )
                + "\n\n"
            )
        return prompt