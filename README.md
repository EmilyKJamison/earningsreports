## ReadMe for earningsreports

### Task

This project is a sample approach to Information Extraction on quarterly corporate earnings 
reports.  We focused on quarterly earnings reports with varied formats.  The varied formats 
make it difficult to extract information with critical high precision.

### About our data

Companies file a 10Q earnings report on a quarterly basis with the SEC.  They also provide 
a press release summary of the 10Q contents.  The 10Q reports follow a standard format and
should be easy to extract data from, but provide no human analysis of content.  The 
summaries provide highlights, short analyses, quotes from CEOs, product descriptions, and 
other helpful material, but are written to be visually appealing and use a wide variety of
formats.  We focus on extracting data from these summary reports.

### Approach

We would like to extract relations (such as <i>quote(WMTCEO, "Our performance exceeded expectations.")<\i> )
from these summaries.  However, the summaries contain a mix of earnings tables, quotes from
persons, phrases stating the positions of a person, graphics, analysis of earnings performance,
product development descriptions, etc.  We could approach relation extraction using supervised 
learned Hearst(1992)-style phrases, such as learning that the pattern "NAME NAME, NOUN and ADJ NOUN"
should be extracted as <i>jobtitle(NAME NAME, NOUN)</i> and <i>jobtitle(NAME NAME, ADJ NOUN)</i>.
However, this method runs that risk that noisy data, such as a mis-parsed table, could 
lower the precision.

We suggest to first train a supervised classifier to label sections of the Summary with 
labels: 

 * PUBINFO publication info, such as contact spokesperson
 * HEADLINE of the document
 * HIGHLIGHTS short bullet points about company performance.  usually 1 sentence each.
 * LONGHIGHLIGHTS paragraph bullet points about company performance
 * PARATEXT paragraph text, i.e., not otherwise formatted
 * TABLE
 * SECTIONHEADER
 * ABOUTCOMPANY boilerplate text about the company, such as founding information and slogans
 * LEGALESE text included for legal reasons, often to counterbalance HIGHLIGHTS
 * FOOTNOTE
 * QUOTE stand-alone text of a quote, often from a high-ranking corporate officer
 * SPEAKER the speaker of a QUOTE
 
After sections are labeled, we expect to have much higher precision with extraction patterns
that are effective with a particular section type.

### Code in this Project

1. We read in all the .pdf Summary text and human annotations in IeSentencePdfReader.
2. We preprocess the text by segmenting sentences and POS-tagging.
3. We extract features such as proportion of long words, percentage of past versus future POS tags, and ngrams.
4. We use a CRF learner to learn a model to label sentences with section labels, and print out results from a 3-fold
CV evaluation experiment.
5. Results are printed, among other places, to stdout for easy reading.

This project extends the UIMA-based DKPro Text Classification framework for easy NLP 
text classification.  

To run the text classification experiment, run RunEarningsSummarySentenceClassifier.


