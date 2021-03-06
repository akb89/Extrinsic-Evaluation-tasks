#! /bin/bash
. $(which env_parallel.bash)
JOBS=1
OUTDIR=/tmp/dl-exe-$(date +%Y-%m-%dT%H%M%S)
PYTHON=python3
TASKS=re,spc,sc,suc,snli
VECTORS=
GITDIR=
OUT=

usage(){
cat << EOF
usage $0 [-j JOBS_NUMBER] [-o OUTPUT_DIRECTORY] [-p PYTHON_EXECUTABLE] [-t TASKS] -v VECTORS_DIRECTORY -g GIT_DIRECTORY -s SCORES_FILE

Run the training for the 5 tasks with j models in parallel

Options
--------------

-v VECTORS_DIRECTORY The directory where the embedding files are
-g GIT_DIRECTORY     The root directory (the git project)
-s SCORES_FILE       TSV file with all the scores
-j JOBS_NUMBER       Number of jobs (parallel jobs) (default 1)
-o OUTPUT_DIRECTORY  The directory where are stored the log files
                     and the intermediate scores. Can be safely
                     remove when everything is finished. (default /tmp/dl-exe-{date})
                     We do not clean up by ourselves
-p PYTHON_EXECUTABLE Path to the python3 executable (default python3)
-t TASKS            Comma separated list of tasks (default = re,spc,sc,suc,snli)
EOF
}

runner() {
    NAME=$(basename $1)
    BASE="${NAME%.*}"

    mkdir -p ${OUTDIR}/${BASE}

    MODEL=${OUTDIR}/${BASE}/model.name.${BASE}.txt

    RE=${OUTDIR}/${BASE}/relation_extraction.${BASE}.txt
    RE_SCORE=${OUTDIR}/${BASE}/relation_extraction.score.${BASE}.txt

    SPC=${OUTDIR}/${BASE}/sentence_polarity_classification.${BASE}.txt
    SPC_SCORE=${OUTDIR}/${BASE}/sentence_polarity_classification.score.${BASE}.txt

    SC=${OUTDIR}/${BASE}/sentiment_classification.${BASE}.txt
    SC_SCORE=${OUTDIR}/${BASE}/sentiment_classification.score.${BASE}.txt

    SNLI=${OUTDIR}/${BASE}/snli.${BASE}.txt
    SNLI_SCORE=${OUTDIR}/${BASE}/snli.score.${BASE}.txt

    SUC=${OUTDIR}/${BASE}/subjectivity_classification.${BASE}.txt
    SUC_SCORE=${OUTDIR}/${BASE}/subjectivity_classification.score.${BASE}.txt

    GLOBAL_SCORES=${OUTDIR}/${BASE}/global_scores.${BASE}.txt

    touch $RE_SCORE
    touch $SPC_SCORE
    touch $SC_SCORE
    touch $SNLI_SCORE
    touch $SUC_SCORE

    echo $BASE > $MODEL

    if echo $TASKS | grep -oi "re"; then
        $PYTHON $GITDIR/Relation_extraction/preprocess.py $1 $GITDIR > /dev/null
        $PYTHON $GITDIR/Relation_extraction/train_cnn.py $1 $GITDIR > $RE
        grep "Accuracy:" $RE | perl -pe "s/Accuracy: //g" | perl -pe "s/ \(max: .+\)//g" > $RE_SCORE
    fi

    if echo $TASKS | grep -oi "spc"; then
        $PYTHON $GITDIR/sentence_polarity_classification/preprocess.py $1 $GITDIR > /dev/null
        $PYTHON $GITDIR/sentence_polarity_classification/train.py $1 $GITDIR > $SPC
        grep "Test-Accuracy:" $SPC | perl -pe "s/Test-Accuracy: //g" > $SPC_SCORE
    fi

    if echo $TASKS | grep -oi "sc"; then
        $PYTHON $GITDIR/sentiment_classification/train.py $1 > $SC
        grep "Test accuracy: " $SC | perl -pe "s/Test accuracy: //g" > $SC_SCORE
    fi

    if echo $TASKS | grep -oi "snli"; then
        $PYTHON $GITDIR/snli/train.py $1 $GITDIR > $SNLI
        grep "Test loss" $SNLI | perl -pe "s/Test loss: .+\/ //g" | perl -pe "s/Test accuracy: //g" > $SNLI_SCORE
    fi

    if echo $TASKS | grep -oi "suc"; then
        $PYTHON $GITDIR/subjectivity_classification/preprocess.py $1 $GITDIR > /dev/null
        $PYTHON $GITDIR/subjectivity_classification/cnn.py $1 $GITDIR > $SUC
        grep "Test-Accuracy:" $SUC | perl -pe "s/Test-Accuracy: //g" > $SUC_SCORE
    fi

    paste $MODEL $RE_SCORE $SPC_SCORE $SC_SCORE $SNLI_SCORE $SUC_SCORE > $GLOBAL_SCORES
    cat $GLOBAL_SCORES
}

while getopts "ht:j:o:p:v:s:g:" OPTION
do
    case $OPTION in
        h)
            usage
            exit 1
            ;;
        j)
            JOBS=$OPTARG
            ;;
        p)
            PYTHON=$OPTARG
            ;;
        o)
            OUTDIR=$OPTARG
            ;;
        g)
            GITDIR=$OPTARG
            ;;
        t)
            TASKS=$OPTARG
            ;;
        v)
            VECTORS=$(find $OPTARG -type f -name '*.txt')
            ;;
        s)
            OUT=$OPTARG
            ;;
    esac
done;

export -f runner
echo -e "MODEL\tRELATION_EXTRACTION\tSENTENCE_POLARITY\tSENTIMENT\tSNLI\tSUBJECTIVITY" > ${OUT}
env_parallel -j ${JOBS} runner ::: ${VECTORS} >> ${OUT}
