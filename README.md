# Term-Translation-with-an-LLM
This repository contains the code and data for Portfolio Task 3, an experiment in term translation with a large language model. It tests whether the wording of a prompt changes how accurately a locally run model translates legal terminology from Chinese into English, using the three prompts set out by Kostikova et al. (2023).

The model is Gemma 4 12B Instruct (QAT), run on a personal computer through LM Studio's local server. The test set is 51 term-dense segments, each taken from a different law in the mainland section of Shaoxing University's Chinese–English parallel corpus. Every segment comes with an official English reference translation, which serves as the standard for terminology.

Each segment was translated three times. Prompt 1 asks only for a translation. Prompt 2 adds a role: a professional translator working on legal documents. Prompt 3 adds a term list that pairs each Chinese term with the English wording the reference uses, written as SOURCE ## TARGET. All three prompts open with the same instruction to return a single translation and nothing else, which keeps alternative renderings, notes and formatting out of the output. Every request was sent as a new single-turn chat with no history, at temperature 0 and with the model's reasoning mode off, so no prompt could see another prompt's answer.

The repository holds the Python script that runs the three passes; a JSON file with the source, reference, three Gemma outputs, a Google Translate output and the term list for every segment; and the zipped corpus section the segments came from. Each output is checked automatically for truncation, extra lines, alternatives, formatting and untranslated Chinese, and all 153 outputs passed.

In this run, the term list raised verbatim agreement with the reference terminology from under half to 94 percent, while the role prompt alone made little difference.

Sun, H. and Yang, J. (2009) Parallel Corpus of China's Legal Documents (PCCLD): Mainland [Chinese–English Parallel Corpus: Law Corpus (Mainland)]. Shaoxing University. Available at: http://corpus.usx.edu.cn/lawcorpus1/ (Accessed: 8 October 2026).

Kostikova, A., Migdisi, K., Szoc, S. and Vanallemeersch, T. (2024) 'Term Translation: Convert or Converse?', in Haddad Haddad, A. and Sosoni, V. (eds) Translating and the Computer 45, Luxembourg, 20–22 November 2023. Geneva: Editions Tradulex, pp. 195–213.
