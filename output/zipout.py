import sys, os, argparse, shutil

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--answerdir", dest="answer_dir", default='output', help="answer directory containing your source files")
    argparser.add_argument("-z", "--zipfile", dest="zipfile", default='output', help="zip file you should upload to Coursys (courses.cs.sfu.ca)")
    opts = argparser.parse_args()

    outputs_zipfile = shutil.make_archive(opts.zipfile, 'zip', opts.answer_dir)
    print("{0} created".format(outputs_zipfile), file=sys.stderr)