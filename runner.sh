#! /bin/bash
. $(which env_parallel.bash)
VECTORS=$(find $1 -type f -name '*.txt')
OUTDIR=$2
GITDIR=$3
OUT=$4
JOBS=$5

runner() {
    BASE=$(basename $1)

    MODEL=${OUTDIR}/model.name.${BASE}.txt

    RE=${OUTDIR}/relation_extraction.${BASE}.txt
    RE_SCORE=${OUTDIR}/relation_extraction.score.${BASE}.txt

    SPC=${OUTDIR}/sentence_polarity_classification.${BASE}.txt
    SPC_SCORE=${OUTDIR}/sentence_polarity_classification.score.${BASE}.txt

    SC=${OUTDIR}/sentiment_classification.${BASE}.txt
    SC_SCORE=${OUTDIR}/sentiment_classification.score.${BASE}.txt

    SNLI=${OUTDIR}/snli.${BASE}.txt
    SNLI_SCORE=${OUTDIR}/snli.score.${BASE}.txt

    SUC=${OUTDIR}/subjectivity_classification.${BASE}.txt
    SUC_SCORE=${OUTDIR}/subjectivity_classification.score.${BASE}.txt

    GLOBAL_SCORES=${OUTDIR}/global_scores.${BASE}.txt

    echo $BASE > $MODEL

    python3 $GITDIR/Relation_extraction/preprocess.py $1 $3 > /dev/null
    python3 $GITDIR/Relation_extraction/train_cnn.py $1 $3 > $RE

    python3 $GITDIR/sentence_polarity_classification/preprocess.py $1 > /dev/null
    python3 $GITDIR/sentence_polarity_classification/train.py $1 > $SPC

    python3 $GITDIR/sentiment_classification/train.py $1 > $SC

    python3 $GITDIR/snli/train.py $1 > $SNLI

    python3 $GITDIR/subjectivity_classification/preprocess.py $1 > /dev/null
    python3 $GITDIR/subjectivity_classification/cnn.py $1 > $SUC

    grep "Accuracy:" $RE | perl -pe "s/Accuracy: //g" | perl -pe "s/ \(max: .+\)//g" > $RE_SCORE
    grep "Test-Accuracy:" $SPC | perl -pe "s/Test-Accuracy: //g" > $SPC_SCORE
    grep "Test accuracy: " $SC | perl -pe "s/Test accuracy: //g" > $SC_SCORE
    grep "Test loss" $SNLI | perl -pe "s/Test loss: .+\/ //g" | perl -pe "s/Test accuracy: //g" > $SNLI_SCORE
    grep "Test-Accuracy:" $SUC | perl -pe "s/Test-Accuracy: //g" > $SUC_SCORE

    paste $MODEL $RE_SCORE $SPC_SCORE $SC_SCORE $SNLI_SCORE $SUC_SCORE > $GLOBAL_SCORES
    cat $GLOBAL_SCORES
}

export -f runner
echo -e "MODEL\tRELATION_EXTRACTION\tSENTENCE_POLARITY\tSENTIMENT\tSNLI\tSUBJECTIVITY" > ${OUT}
env_parallel -j ${JOBS} runner ::: ${VECTORS} >> ${OUT}
