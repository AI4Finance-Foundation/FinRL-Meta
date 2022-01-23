def render_to_file(**kwargs):
    log_header = kwargs.get("log_header", False)
    log_filename = kwargs.get("log_filename", "./data/log/log_")
    printout = kwargs.get("printout", False)
    balance = kwargs.get("balance")
    balance_initial = kwargs.get("balance_initial")
    tranaction_close_this_step = kwargs.get("tranaction_close_this_step", [])
    done_information = kwargs.get("done_information", "")
    profit = balance - balance_initial
    tr_lines = ""
    tr_lines_comma = ""
    _header = ''
    _header_comma = ''
    if log_header:
        _header = f'{"Ticket":>8}{"Symbol":8}{"Type":8}{"ActionTime":>20} \
                            {"ActionPrice":14}{"MaxDD":8}{"CloseTime":>20}{"ClosePrice":14} \
                            {"Reward":8}{"SL":8}{"PT":8}{"DateDuration":20}{"Status":8}\n'

        _header_comma = 'Ticket,Symbol,Type,ActionTime,ActionPrice,MaxDD,CloseTime,ClosePrice,Reward,SL,PT,DateDuration,Status\n'

    if tranaction_close_this_step:
        for _tr in tranaction_close_this_step:
            tr_lines += f'{_tr["Ticket"]:>8} {_tr["Symbol"]:8} {_tr["Type"]:>4} {_tr["ActionTime"]:16} \
                {_tr["ActionPrice"]:6.5f} {_tr["MaxDD"]:8} {_tr["CloseTime"]:16} {_tr["ClosePrice"]:6.5f} \
                {_tr["Reward"]:4.0f} {_tr["SL"]:4.0f} {_tr["PT"]:4.0f} {_tr["DateDuration"]:20} {_tr["Status"]:8}\n'

            tr_lines_comma += f'{_tr["Ticket"]},{_tr["Symbol"]},{_tr["Type"]},{_tr["ActionTime"]}, \
                {_tr["ActionPrice"]:6.5f},{_tr["MaxDD"]},{_tr["CloseTime"]},{_tr["ClosePrice"]:6.5f}, \
                {_tr["Reward"]:4.0f},{_tr["SL"]:4.0f},{_tr["PT"]:4.0f},{_tr["DateDuration"]},{_tr["Status"]}\n'

    log = _header_comma + tr_lines_comma
    # log = f"Step: {current_step}   Balance: {balance}, Profit: {profit} \
    #     MDD: {max_draw_down_pct}\n{tr_lines_comma}\n"
    if done_information:
        log += done_information
    if log:
        with open(log_filename, 'a+') as _f:
            _f.write(log)
            _f.close()

    tr_lines += _header
    if printout and tr_lines:
        print(tr_lines)
        if done_information:
            print(done_information)
