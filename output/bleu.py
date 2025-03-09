import sys, re
import sacrebleu
import json

bleu = sacrebleu.metrics.BLEU(effective_order=True)

def compute_bleu(references, output_data):
    bleu_score = 0.0
    if len(references) == len(output_data):
        score = 0.0
        total = 0.0
        len_r = 0.0
        len_h = 0.0
        unscored = 0.0
        min_bleu, max_bleu = 100, 0
        i = 0
        for (ref, output) in zip(references, output_data):
            ref_dict  = json.loads(ref)
            output_dict = json.loads(output)
            r = [ref_dict['output']]
            if '' in r:
                # print(r)
                unscored += 1
                continue
            h = output_dict['output']
            if h == '':
                # print(h)
                unscored += 1
                continue
            len_r += len(r[0].split(' '))
            len_h += len(h.split(' '))
            bss = bleu.sentence_score(h, r).score
            score += bss
            if bss > max_bleu:
                max_bleu = bss
                max_id = i
            if bss < min_bleu and bss > 50:
                min_bleu = bss
                min_id = i
            # print(score)
            total += 1.
            i += 1
        bleu_score = score / total
        avg_len_r = len_r / total
        avg_len_h = len_h / total
    print(f'BLEU:{bleu_score:.4f}, Ref words:{avg_len_r:.1f}, Ans words:{avg_len_h:.1f}, Scored: {total:.0f}/{len(references)-150:.0f}')    
    # print(max_id, min_id)
    # print(max_bleu, min_bleu)
    return bleu_score

if __name__ == '__main__':
    import os, argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    argparser.add_argument("-o", "--outputfile", dest="output", default='output.txt', help="output file created by chunker.py [default: output.txt]")
    opts = argparser.parse_args()

    references = {}
    ref_data = []
    output_data = []
    with open(opts.ref, 'r') as ref:
        ref_data = list(filter(lambda k: k, [str(x) for x in ref.read().splitlines()]))
        for line in ref_data:
            src_id, _, suggested_reference = line.split('||')
            references.setdefault(src_id, [])
            references[src_id].append(suggested_reference)
    with open(opts.output) as out:
        output_data = list(filter(lambda k: k, [str(x) for x in out.read().splitlines()]))
        output_data = [line.split('||') for line in output_data]
        output_data = output_data[:len(ref_data)]
    print(f"bleu score: {compute_bleu(references, output_data)}")
