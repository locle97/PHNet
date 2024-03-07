def log(msg, lvl="info"):
    if lvl == "info":
        print(f"***********{msg}****************")
    if lvl == "error":
        print(f"!!! Exception: {msg} !!!")
