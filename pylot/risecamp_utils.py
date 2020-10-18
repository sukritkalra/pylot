import inspect

def validate_problem_1(op):
    add_callback_line = list(filter(lambda s: s.find('add_callback') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.__init__)[0])))

    if len(add_callback_line) == 0:
        failure_reason = "add_callback was not called!"
        return True, failure_reason

    if len(add_callback_line) != 1:
        failure_reason = "add_callback was called {} times! Make sure there is only one add_callback_invocation".format(len(add_callback_line))
        return True, failure_reason

    add_callback_line = add_callback_line[0].strip() 
    arguments = add_callback_line[add_callback_line.find("(")+1:add_callback_line.find(")")].split(',', maxsplit=1)

    if len(arguments) == 0:
        failure_reason = "add_callback was not invoked with the callback and the output stream."
        return True, failure_reason

    if arguments[0] != "self.on_message":
        failure_reason = "self.on_message was not registered with the camera_stream"
        return True, failure_reason

    if len(arguments) == 1:
        failure_reason = "Only a single argument was passed to the add_callback method. Did you forget the output stream?" 
        return True, failure_reason

    second_argument = arguments[1].strip()

    if not second_argument.startswith('[') or not second_argument.endswith(']'):
        failure_reason = "The second argument to add_callback was not a list! " \
        "Make sure you're invoking add_callback with the list of output stream(s)"
        return True, failure_reason

    list_args = list(map(lambda a: a.strip(), second_argument[1:-1].split(',')))
    if len(list_args) != 1:
        failure_reason = "More than one output stream was passed to add_callback. " \
        "Make sure to only pass the detected_objects_stream."
        return True, failure_reason

    if list_args[0] != "detected_objects_stream":
        failure_reason = "The detected_objects_stream was not passed as the second argument to add_callback"
        return True, failure_reason


    # Check errors in on_message.
    detect_objects_line = list(filter(lambda s: s.find('detect_objects') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.on_message)[0])))

    if len(detect_objects_line) == 0:
        failure_reason = "detect_objects was not called!"
        return True, failure_reason

    if len(detect_objects_line) != 1:
        failure_reason = "detect_objects was called {} times! Make sure that there is only invocation of detect_objects".format(len(detect_objects_line))
        return True, failure_reason

    detect_objects_line = detect_objects_line[0].strip()
    saved_variable = detect_objects_line.split('=')[0].strip()
    arguments = detect_objects_line[detect_objects_line.find("(")+1:detect_objects_line.find(")")].split(',', maxsplit=1)

    if len(arguments) != 1:
        failure_reason = "detect_objects was invoked with {} argument(s)! Make sure that it is only invoked with message.".format(len(arguments))
        return True, failure_reason
        
    if arguments[0] != "message":
        failure_reason = "detect_objects was not invoked with the message."
        return True, failure_reason

    # Check errors in send.
    send_line = list(filter(lambda s: s.find('send') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.on_message)[0])))

    if len(send_line) == 0:
        failure_reason = "send was not called!"
        return True, failure_reason

    if len(send_line) != 1:
        failure_reason = "send was called {} times! Make sure that there is only one invocation of send".format(len(send_line))
        return True, failure_reason

    send_line = send_line[0].strip()
    arguments = send_line[send_line.find("(")+1:send_line.find(")")].split(",", maxsplit=1)

    if len(arguments) == 1 and arguments[0] == '':
        failure_reason = "send was invoked with no arguments! Make sure that it is only invoked with detected_objects."
        return True, failure_reason

    if len(arguments) > 1:
        failure_reason = "send was invoked with {} argument(s)! Make sure that it is only invoked with detected_objects.".format(len(arguments))
        return True, failure_reason

    if arguments[0] != saved_variable:
        failure_reason = "send was not invoked with {}".format(saved_variable)
        return True, failure_reason

    return False, ""
