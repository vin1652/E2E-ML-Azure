import sys 

def error_message_detail(error, error_detail: sys):
    """
    Extracts the error message from the exception and returns it.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script: [{file_name}] at line number: [{exc_tb.tb_lineno}] with message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It overrides the constructor to provide a detailed error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    

    