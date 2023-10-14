#!/usr/bin/env python
import json
import fire


def main(input_file, output_file):
    with open(input_file) as i_:
        dataset = json.load(i_)

    processed = []
    for data in dataset['data']:
        title = data['title']
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            idx = context.find('[SEP]') + len('[SEP]')
            context = context[idx:]
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]
                processed.append(dict(
                    title=title,
                    context=context,
                    question=question,
                    answer=answer
                ))
    
    with open(output_file, 'w') as o_:
        json.dump(processed, o_, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)