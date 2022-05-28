class RemoteImageException(Exception):
    def __init__(self, status_code, message=None):
        self.status_code = status_code
        if status_code == 404:
            message = 'Remote image not found'
        super().__init__(message)
