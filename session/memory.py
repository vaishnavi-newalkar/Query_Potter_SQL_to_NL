# class ConversationMemory:
#     """
#     Manages conversational memory for a session.
#     """

#     def __init__(self, max_turns: int = 10):
#         self.max_turns = max_turns
#         self.history = []  # list of {role, content}

#     def add_user_message(self, content: str):
#         self._add_message("user", content)

#     def add_system_message(self, content: str):
#         self._add_message("system", content)

#     def _add_message(self, role: str, content: str):
#         if not content:
#             return

#         self.history.append({
#             "role": role,
#             "content": content
#         })

#         # Sliding window: keep last N turns
#         if len(self.history) > self.max_turns * 2:
#             self.history = self.history[-self.max_turns * 2:]

#     def get_history(self):
#         return self.history

#     def clear(self):
#         self.history = []


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    """
    Manages conversational memory for a session.
    """

    def __init__(self):
        self.history = ChatMessageHistory()

    def add_user_message(self, content: str):
        self.history.add_user_message(content)

    def add_system_message(self, content: str):
        self.history.add_ai_message(content)

    def add_turn(self, user_input: str, ai_output: str):
        self.history.add_user_message(user_input)
        self.history.add_ai_message(ai_output)

    def get_history(self, k: int = 5):
        messages = self.history.messages[-(k * 2):]
        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                history.append(f"AI: {message.content}")

        return "\n".join(history)

    def clear(self):
        self.history.clear()