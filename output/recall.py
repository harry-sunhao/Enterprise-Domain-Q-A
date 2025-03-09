def compute_recall(ref_data, output_data):
    hit = 0.
    rhit = 0.
    na_hit = 0.
    ab_hit = 0.
    rab_hit = 0.
    total = 0.
    na_total = 0.
    ab_total = 0.
    for (ref, output) in zip(ref_data, output_data):
        ref_id  = ref.split('-')[0]
        output_list = output.split(',')
        # if '' in output_list:
        #     print(output_list)
        if ref_id == '':
            na_total += 1
        else:
            ab_total += 1
            
        if ref_id in output_list:
            if ref_id == '':
                na_hit += 1
                hit += 1
                rhit += 1
            else:
                ab_hit += 1
                rab_hit += 1 / len(output_list)
                hit += 1
                rhit += 1 / len(output_list)
        total += 1
    Recall_score = 100 * hit / total
    rRecall_score = 100 * rhit / total
    na_recall = 100 * na_hit / na_total
    ab_recall = 100 * ab_hit / ab_total
    rab_recall = 100 * rab_hit / ab_total
    if total == na_total + ab_total:
        print(f'Recall: {Recall_score:.4f}, rRecall: {rRecall_score:.4f}; A-Recall: {ab_recall:.4f}, A-rRecall: {rab_recall:.4f}; NA-Recall: {na_recall:.4f}')
    return (Recall_score)