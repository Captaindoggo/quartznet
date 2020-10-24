import jiwer



def decoder(seq, id_dict, emp_token = 1):
    res = []
    res_decod = []
    ctr = 0
    win = []
    win_decod = []
    for i in seq:
        if i == emp_token:
            res.extend(win)
            res_decod.extend(win_decod)
            win = []
            win_decod = []
            ctr = 0
        else:
            if ctr == 0:
                win.append(i)
                win_decod.append(id_dict[int(i)])
                ctr+=1
            else:
                if win[ctr-1] != i:
                    win.append(i)
                    win_decod.append(id_dict[int(i)])
                    ctr+=1
    if seq[-1] != emp_token:
        res.extend(win)
        res_decod.extend(win_decod)
    return res, ''.join(res_decod)

def ans_decoder(seq, id_dict, space=0):
    decoded = []
    pad = 0
    for i in seq:
        if i == space:
            pad += 1
        else:
            if pad != 0:
                decoded.append(id_dict[space])
                pad=0
                decoded.append(id_dict[int(i)])
            else:
                decoded.append(id_dict[int(i)])
    return ''.join(decoded)


def decode_wer(pred, truth, id_dict, space=0, emp_t=1, is_list = True, verbose = 2):
    decoded_pred = []
    wer = 0
    n=0
    pr = 0
    if is_list:
        for j in range(len(pred)):
            pred_batch = pred[j]
            truth_batch = truth[j]
            for i in range(len(pred_batch)):
                clean_pred, text_pred = decoder(pred_batch[i], id_dict, emp_token=emp_t)
                text_truth = ans_decoder(truth_batch[i], id_dict, space=space)
                if len(text_truth) != 0 and text_truth != ' ':
                    wer_1 = jiwer.wer(text_truth, text_pred)
                    wer += wer_1
                    if pr < verbose:
                        print('text pred:', text_pred, ',  text true:', text_truth, ',  wer:', wer_1)
                        pr +=1
                    decoded_pred.append(text_pred)
                    n+=1
    else:
        for i in range(len(pred)):
                clean_pred, text_pred = decoder(pred[i], id_dict, emp_token=emp_t)
                text_truth = ans_decoder(truth[i], id_dict, space=space)
                if len(text_truth) != 0 and text_truth != ' ':
                    wer_1 = jiwer.wer(text_truth, text_pred)
                    wer += wer_1
                    if pr < verbose:
                        print('text pred:', text_pred, 'text true:', text_truth)
                        pr +=1
                    decoded_pred.append(text_pred)
                    n+=1
    return 100*(wer/n)