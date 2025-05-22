#!/usr/bin/env python3
import sys
import csv

def main(raw_args):
    langs = set(raw_args)
    langs.add('en')
    reader = csv.reader(sys.stdin, delimiter='\t')

    def pred(row):
        head = row[2]
        tail = row[3]
        head_lang = head.split('/')[2]
        tail_lang = tail.split('/')[2]
        return (head_lang == 'en' and (tail_lang == 'en' or (tail_lang in langs and row[1] == '/r/Synonym'))) or \
                (tail_lang == 'en' and (head_lang == 'en' or (head_lang in langs and row[1] == '/r/Synonym')))

    to_keep = filter(pred, reader)
    to_keep = map(lambda row: row[:4], to_keep)
    writer = csv.writer(sys.stdout, delimiter='\t')
    writer.writerows(to_keep)

if __name__ == "__main__":
    main(sys.argv[1:])