from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np


class Metrics:
    """
    텍스트 생성 평가 지표
    """

    def __init__(self):
        """
        TODO: 초기화
        - ROUGE scorer 생성
        """
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],  # ← 구체적으로!
                                              use_stemmer=True
                                              )

    def compute_bleu(self, references, hypotheses, max_n=4):
        """
        BLEU score 계산

        Args:
            references: List[List[str]] - 정답 문장들 (tokenized)
            references = [[['나는', '방금', '빨간', '사과를', '먹었다'], ['나는', '사과를', '먹음']],
                          [['창밖을', '보니', '하늘이', '파랗다']]
                         ]
            hypotheses: List[str] - 생성된 문장들 (tokenized)
            hypotheses = [['나는', '오늘', '사과를', '먹었다'],
                          ['하늘은', '매우', '푸르다']
                         ]
            max_n: BLEU-N (1, 2, 3, 4)
            각 hypotheses에 대한 reference가 여러 개일 수 있음. hypotheses에 대한 references중에 bleu가 가장 높은 값을 사용함.
        Returns:
            dict: {'bleu-1': ..., 'bleu-2': ..., ...}
        """
        # TODO: 구현
        # - BLEU-1, BLEU-2, BLEU-3, BLEU-4 계산
        # - corpus_bleu 사용
        bleu_dict = {}
        for n in range(1, max_n+1):
            weights = tuple([1.0/n] * n + [0.0] * (4 - n))
            bleu_score = corpus_bleu(references, hypotheses,weights=weights)
            bleu_dict[f'bleu-{n}'] = bleu_score
        return bleu_dict

    def compute_rouge(self, references, hypotheses):
        """
        ROUGE score 계산

        Args:
            references: List[str] - 정답 문장들
            hypotheses: List[str] - 생성된 문장들

        Returns:
            dict: {'rouge-1': ..., 'rouge-2': ..., 'rouge-l': ...}
        """
        # TODO: 구현
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }

        for reference, hypothesis in zip(references, hypotheses):
            # 문자열 변환
            if isinstance(reference, list):
                reference = ' '.join(reference)
            if isinstance(hypothesis, list):
                hypothesis = ' '.join(hypothesis)

            scores = self.rouge.score(reference, hypothesis)

            # 각 metric과 측정치 저장
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[metric]['precision'].append(
                    scores[metric].precision
                )
                rouge_scores[metric]['recall'].append(
                    scores[metric].recall
                )
                rouge_scores[metric]['fmeasure'].append(
                    scores[metric].fmeasure
                )

        # 평균 계산 (F-measure만 반환)
        return {
            'rouge-1': np.mean(rouge_scores['rouge1']['fmeasure']),
            'rouge-2': np.mean(rouge_scores['rouge2']['fmeasure']),
            'rouge-l': np.mean(rouge_scores['rougeL']['fmeasure'])
        }

    def compute_meteor(self, references, hypotheses):
        """
        METEOR score 계산

        Args:
            references: List[str] - 정답 문장들
            hypotheses: List[str] - 생성된 문장들

        Returns:
            float: METEOR score
        """
        # TODO: 구현 (선택)
        pass

    def compute_all(self, references, hypotheses):
        """
        모든 지표 계산

        Args:
            references: List[str] or List[List[str]]
            hypotheses: List[str]

        Returns:
            dict: 모든 metric 결과
        """
        # TODO: 구현
        # - BLEU, ROUGE 모두 계산
        # - 하나의 dict로 반환
        results = {}

        # BLEU (토큰화된 형태 필요)
        # references를 [[ref]] 형태로 변환
        refs_for_bleu = []
        hyps_for_bleu = []

        for ref, hyp in zip(references, hypotheses):
            # 문자열이면 split
            if isinstance(ref, str):
                ref = ref.split()
            if isinstance(hyp, str):
                hyp = hyp.split()

            refs_for_bleu.append([ref])  # [[tokens]]
            hyps_for_bleu.append(hyp)  # [tokens]

        # BLEU 계산
        bleu_scores = self.compute_bleu(refs_for_bleu, hyps_for_bleu)
        results.update(bleu_scores)

        # ROUGE 계산
        rouge_scores = self.compute_rouge(references, hypotheses)
        results.update(rouge_scores)

        return results
# 테스트
if __name__ == '__main__':
    metrics = Metrics()

    # 테스트 데이터
    # references = [
    #     [["the", "quick", "brown", "fox"]],  # 주의: [[...]] 형태!
    #     [["hello", "test"],['hello','world']]
    # ]
    #
    # hypotheses = [
    #     ["the", "quick", "brown","fox"],
    #     ["hello", "world"]
    # ]
    #
    # # BLEU 테스트
    # bleu_scores = metrics.compute_bleu(references, hypotheses)
    # print("BLEU Scores:")
    # for metric, score in bleu_scores.items():
    #     print(f"  {metric}: {score:.4f}")

    references = ['the quick brown fox', 'hello computer world']
    hypotheses = ['the quick brown fox', 'hello world']
    # rouge_scores = metrics.compute_rouge(references, hypotheses)
    # print(rouge_scores)
    scores = metrics.compute_all(references, hypotheses)
    print(scores)