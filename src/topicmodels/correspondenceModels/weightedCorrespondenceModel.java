package topicmodels.correspondenceModels;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import LBFGS.LBFGS;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc4DCM;
import structures._SparseFeature;
import structures._Stn;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

/**
 * Created by jetcai1900 on 12/17/16.
 */
public class weightedCorrespondenceModel extends LDA_Variational {

    protected double[][] m_beta;
    protected double[] m_alpha_c;
    protected double[] m_alpha_stat;
    protected double[] m_alpha_c_stat;
    protected double[][] m_beta_stat;
    protected int m_parentDocNum;
    protected int m_childDocNum;
    protected double m_lbfgsConverge;

    public weightedCorrespondenceModel(int number_of_iteration, double converge, double beta,
                                       _Corpus c, double lambda, int number_of_topics, double alpha,
                                       int varMaxIter, double varConverge, double lbfgsConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_varConverge = varConverge;
        m_varMaxIter = varMaxIter;
        m_logSpace = true;
        m_lbfgsConverge = lbfgsConverge;

    }

    @Override
    public void createSpace(){
        super.createSpace();

        m_parentDocNum = 0;
        m_childDocNum = 0;

        m_beta = new double[number_of_topics][vocabulary_size];
        m_alpha_c  = new double[number_of_topics];

        for(int k=0; k<number_of_topics; k++) {
            m_alpha_c[k] = d_alpha;
            Arrays.fill(m_beta[k], d_beta);
        }

    }

    @Override
    public String toString(){
        return String.format("WCM, Variational Inference[k:%d, alpha:%.2f, beta:%.2f]",number_of_topics, d_alpha, d_beta);
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection){
        init();
        for(_Doc d:collection){
            if(d instanceof _ParentDoc4DCM){
                int totalWords = 0;
                double totalLambda = 0;

                m_parentDocNum += 1;
                _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
                pDoc.setTopics4Variational(number_of_topics, d_alpha, vocabulary_size, d_beta);

                totalWords += pDoc.getTotalDocLength();
                for(_Stn stnObj:pDoc.getSentences())
                    stnObj.setTopicsVct(number_of_topics);

                for(_ChildDoc cDoc:pDoc.m_childDocs){
                    totalWords += cDoc.getTotalDocLength();
                    m_childDocNum += 1;
                    cDoc.setTopics4Variational(number_of_topics, d_alpha);

                    //update the article thread sufficient statistics
                    for(int n=0; n<cDoc.getSparse().length; n++){
                        _SparseFeature fv = cDoc.getSparse()[n];
                        int wID = fv.getIndex();
                        double wVal = fv.getValue();
                        for(int k=0; k<number_of_topics; k++){
                            pDoc.m_lambda_stat[k][wID] += cDoc.m_phi[n][k]*wVal;
                        }
                    }
                }

                for(int k=0; k<number_of_topics; k++) {
                    pDoc.m_lambda_topicStat[k] = Utils.sumOfArray(pDoc.m_lambda_stat[k]);
                    totalLambda += pDoc.m_lambda_topicStat[k];
                }

//                System.out.println("totalWords\t"+totalWords+"\t"+totalLambda);
            }
        }
        imposePrior();
    }

    @Override
    protected void init(){
        m_alpha_stat = new double[number_of_topics];
        m_alpha_c_stat = new double[number_of_topics];
        m_beta_stat = new double[number_of_topics][vocabulary_size];

        for(int k=0; k<number_of_topics; k++){
            m_alpha_stat[k] = 0;
            m_alpha_c_stat[k] =0;

            Arrays.fill(m_beta_stat[k], 0);
        }
    }

    @Override
    public void EM() {
        System.out.format("Starting %s...\n", toString());

        long starttime = System.currentTimeMillis();

        m_collectCorpusStats = true;
        initialize_probability(m_trainSet);

//		double delta, last = calculate_log_likelihood(), current;
        double delta=0, last=0, current=0;
        int i = 0, displayCount = 0;
        do {
            init();

            for(_Doc d:m_trainSet)
                calculate_E_step(d);

            calculate_M_step(i);

            if (m_converge>0 || (m_displayLap>0 && i%m_displayLap==0 && displayCount > 6)){//required to display log-likelihood
                current = calculate_log_likelihood();//together with corpus-level log-likelihood
//				current += calculate_log_likelihood();//together with corpus-level log-likelihood

                if (i>0)
                    delta = (last-current)/last;
                else
                    delta = 1.0;
                last = current;
            }

            if (m_converge>0) {
                System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
                infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
            }

            if (m_converge>0 && Math.abs(delta)<m_converge)
                break;//to speed-up, we don't need to compute likelihood in many cases
        } while (++i<this.number_of_iteration);

        finalEst();

        long endtime = System.currentTimeMillis() - starttime;
        System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
        infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
    }

    @Override
    protected void imposePrior() {
        if (word_topic_prior != null) {
            for (int k = 0; k < number_of_topics; k++) {
                for (int v = 0; v < vocabulary_size; v++) {
                    m_beta[k][v] = word_topic_prior[k][v];//how could we make sure that beta is not zero
                }
            }
        }
    }

    @Override
    public void LoadPrior(String fileName, double eta) {
        if (fileName == null || fileName.isEmpty())
            return;

        try {
            if (word_topic_prior == null)
                word_topic_prior = new double[number_of_topics][vocabulary_size];
            else {
	            for (int k = 0; k < number_of_topics; k++)
	                Arrays.fill(word_topic_prior[k], 0);
            }

            String tmpTxt;
            String[] lineContainer;
            String[] featureContainer;
            int tid = 0;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for (int i = 0; i < m_corpus.getFeatureSize(); i++)
                featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));

            while ((tmpTxt = br.readLine()) != null) {
                tmpTxt = tmpTxt.trim();
                if (tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");

                tid = Integer.parseInt(lineContainer[0]);
                for (int i = 1; i < lineContainer.length; i++) {
                    featureContainer = lineContainer[i].split(":");

                    String featureName = featureContainer[0];
                    double featureProb = Double.parseDouble(featureContainer[1]);

                    int featureIndex = featureNameIndex.get(featureName);

                    word_topic_prior[tid][featureIndex] = featureProb;
                }
            }

            System.out.println("prior is added");
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double calculate_E_step(_Doc d){
        if(d instanceof _ChildDoc)
            return 0;

        _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;

        double last = 1;
        if(m_varConverge>0)
            last = calculate_log_likelihood(pDoc);

        double current = last, converge, logSum, wVal;
        int iter=0, wID;

        do{
            updateEta4Parent(pDoc);
            updateGamma4Parent(pDoc);
//            updateLambda(pDoc);
            
            updatePi4Child(pDoc);
            updateZeta4Child(pDoc);
            updateEta4Child(pDoc);
            updateLambda(pDoc);

            if(m_varConverge>0){
                current = calculate_log_likelihood(pDoc);
                converge = Math.abs((current-last)/last);
                last = current;

                if(converge<m_varConverge)
                    break;
            }
        } while(++iter<m_varMaxIter);

        collectStats(pDoc);

        return current;
    }

    protected void collectStats(_ParentDoc4DCM pDoc){

        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        for(int k=0; k<number_of_topics; k++) {
            m_alpha_stat[k] += Utils.digamma(pDoc.m_sstat[k]) - Utils.digamma(gammaSum);
            double lambdaSum = Utils.sumOfArray(pDoc.m_lambda_stat[k]);
            for(int v=0; v<vocabulary_size; v++){
                m_beta_stat[k][v] += Utils.digamma(pDoc.m_lambda_stat[k][v])-Utils.digamma(lambdaSum);
            }
        }

        for(_ChildDoc cDoc:pDoc.m_childDocs) {

            double piSum = Utils.sumOfArray(cDoc.m_sstat);
            for (int k = 0; k < number_of_topics; k++)
                m_alpha_c_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);
        }
    }

    public void updateEta4Parent(_ParentDoc4DCM pDoc){
        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                pDoc.m_phi[n][k] = Utils.digamma(pDoc.m_sstat[k]) + Utils.digamma(pDoc.m_lambda_stat[k][wID]);
                pDoc.m_phi[n][k] -= Utils.digamma(pDoc.m_lambda_topicStat[k]);
            }

            double logSum = logSumOfExponentials(pDoc.m_phi[n]);
            double phiSum = 0;

            for(int k=0; k<number_of_topics; k++){
                if((pDoc.m_phi[n][k]-logSum)<-200){
                    pDoc.m_phi[n][k] = 1e-20;
                }else {
                    pDoc.m_phi[n][k] = Math.exp(pDoc.m_phi[n][k] - logSum);
                }
                phiSum += pDoc.m_phi[n][k];
            }

            if(Math.abs(phiSum-1)>0)
                System.out.println("phiSum for article\t" + phiSum);
        }
    }

    public void updateGamma4Parent(_ParentDoc4DCM pDoc){
        int[] iflag = {0}, iprint={-1,3};
        double fValue = 0;
        int fSize = pDoc.m_sstat.length;

        double[] gammaGradient = new double[fSize];
        Arrays.fill(gammaGradient,0);

        double[] gammaDiag = new double[fSize];
        Arrays.fill(gammaDiag, 0);

        double[] gamma = new double[fSize];
        double[] oldGamma = new double[fSize];

        for(int k=0; k<fSize; k++){
            gamma[k] = Math.log(pDoc.m_sstat[k]);
            oldGamma[k] = Math.log(pDoc.m_sstat[k]);
        }

//        double diff = 0;

        try{
            do{
                double diff = 0;
                fValue = gammaFuncGradientVal(pDoc, gamma, gammaGradient);
                LBFGS.lbfgs(fSize, 4, gamma, fValue, gammaGradient, false, gammaDiag, iprint, 1e-2, 1e-10, iflag);

                for(int k=0; k<fSize; k++){
                    double tempDiff = 0;
                    tempDiff = gamma[k]-oldGamma[k];
                    if(Math.abs(tempDiff)>diff){
                        diff = Math.abs(tempDiff);
                    }
                    oldGamma[k] = gamma[k];
                }

                if(diff<m_lbfgsConverge){
//                    System.out.print("diff\t"+diff+"finish update gamma");
                    break;
                }

            }while(iflag[0]!=0);
        }catch(LBFGS.ExceptionWithIflag e){
            e.printStackTrace();
        }

        for(int k=0; k<fSize; k++){
            pDoc.m_sstat[k] = Math.exp(gamma[k]);

//            System.out.println(pDoc.getName()+"\tpDoc.m_sstat[]"+pDoc.m_sstat[k]);
        }

    }

    public double gammaFuncGradientVal(_ParentDoc4DCM pDoc, double[]gamma, double[] gammaGradient){

        Arrays.fill(gammaGradient, 0);
        double funcVal = 0;

        double[] expGamma = new double[number_of_topics];
        double expGammaSum = 0;

        for(int k=0; k<number_of_topics; k++){
            expGamma[k] = Math.exp(gamma[k]);
            expGammaSum += expGamma[k];
        }

        funcVal -= Utils.lgamma(expGammaSum);

        double constantGradient = 0;
        for(int k=0; k<number_of_topics; k++){
            constantGradient += (m_alpha[k]-expGamma[k])*Utils.trigamma(expGammaSum);
        }

        for(int k=0; k<number_of_topics; k++){
            gammaGradient[k] = (m_alpha[k]-expGamma[k])*Utils.trigamma(expGamma[k]);
            funcVal += (m_alpha[k]-expGamma[k])*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));
            funcVal += Utils.lgamma(expGamma[k]);
        }

        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                funcVal += pDoc.m_phi[n][k]*wVal*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));

                gammaGradient[k] += pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGamma[k]);
                constantGradient += pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
//                gammaGradient[k] -= pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
            }
        }


        for(_ChildDoc cDoc:pDoc.m_childDocs){

            double piSum = Utils.sumOfArray(cDoc.m_sstat);

            _SparseFeature[] cDocFvs = cDoc.getSparse();
            for(int n=0; n<cDocFvs.length; n++) {
                int wID = cDocFvs[n].getIndex();
                double wVal = cDocFvs[n].getValue();
                for(int k=0; k<number_of_topics; k++){
                    funcVal += cDoc.m_phi[n][k]*wVal*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));
                    funcVal -= cDoc.m_phi[n][k]*wVal*(Utils.dotProduct(cDoc.m_sstat, expGamma))/(piSum*expGammaSum*cDoc.m_zeta);

                    gammaGradient[k] += cDoc.m_phi[n][k]*wVal*Utils.trigamma(expGamma[k]);
                    constantGradient += cDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
//                    gammaGradient[k] -= cDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
                    double temp = cDoc.m_sstat[k]*expGammaSum-Utils.dotProduct(cDoc.m_sstat, expGamma);
                    gammaGradient[k] -= cDoc.m_phi[n][k]*wVal*temp/(piSum*expGammaSum*expGammaSum*cDoc.m_zeta);
                }

            }
        }

        for(int k=0; k<number_of_topics; k++){
            gammaGradient[k] -= constantGradient;
            gammaGradient[k] *= expGamma[k];
            gammaGradient[k] = 0-gammaGradient[k];
        }

        return -funcVal;
    }

    public void updateLambda(_ParentDoc4DCM pDoc){
       _SparseFeature[] fvs = pDoc.getSparse();

        int totalWord = 0;
        double totalLambda = 0;

        for(int k=0; k<number_of_topics; k++)
            for(int v=0; v<vocabulary_size; v++)
                pDoc.m_lambda_stat[k][v] = m_beta[k][v];

        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            totalWord += wVal;

            double phiSum = Utils.sumOfArray(pDoc.m_phi[n]);
            if(Math.abs(phiSum-1)>1){
                System.out.println("inequal to 1\t"+n+phiSum);
            }

            for(int k=0; k<number_of_topics; k++){
                pDoc.m_lambda_stat[k][wID] += wVal*pDoc.m_phi[n][k];

                if(Double.isNaN(pDoc.m_lambda_stat[k][wID])){
                    System.out.println("nan error article\t"+n+" "+wID);
                }
            }
        }

        for(_ChildDoc cDoc:pDoc.m_childDocs){
            _SparseFeature[] cFvs = cDoc.getSparse();
            for(int n=0; n<cFvs.length; n++){
                int wID = cFvs[n].getIndex();
                double wVal = cFvs[n].getValue();

                totalWord += wVal;

                double phiSum = Utils.sumOfArray(cDoc.m_phi[n]);
                if(Math.abs(phiSum-1)>1){
                    System.out.println("inequal to 1\t"+n+phiSum);
                    for(int k=0; k<number_of_topics; k++){
                        System.out.println("\t\t"+cDoc.m_phi[n][k]);
                    }
                }

                for(int k=0; k<number_of_topics; k++){
                    pDoc.m_lambda_stat[k][wID] += wVal*cDoc.m_phi[n][k];

                    if(Double.isNaN(pDoc.m_lambda_stat[k][wID])){
                        System.out.println("nan error comment\t"+n+" "+wID);
                    }
                }
            }
        }

        for(int k=0; k<number_of_topics; k++) {
            pDoc.m_lambda_topicStat[k] = Utils.sumOfArray(pDoc.m_lambda_stat[k]);
            totalLambda += pDoc.m_lambda_topicStat[k];
        }

//        System.out.println("total Words in this doc\t"+pDoc.getName()+"\t"+totalWord+"\t"+totalLambda);
    }

    public void updatePi4Child(_ParentDoc4DCM pDoc){
        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        for(_ChildDoc cDoc:pDoc.m_childDocs) {
            int[] iflag = {0}, iprint = {-1, 3};
            double fValue = 0;
            int fSize = cDoc.m_sstat.length;

            double[] piGradient = new double[fSize];
            Arrays.fill(piGradient, 0);

            double[] piDiag = new double[fSize];
            Arrays.fill(piDiag, 0);

            double[] pi = new double[fSize];
            double[] oldPi = new double[fSize];

            for(int k=0; k<fSize; k++){
                pi[k] = Math.log(cDoc.m_sstat[k]);
                oldPi[k] = Math.log(cDoc.m_sstat[k]);
            }

            try {
                do {
                    double diff = 0;

                    fValue = piFuncGradientVal(pDoc, gammaSum, cDoc, pi, piGradient);
                    LBFGS.lbfgs(fSize, 4, pi, fValue, piGradient, false, piDiag, iprint, 1e-2, 1e-10, iflag);

                    for(int k=0; k<fSize; k++){
                        double tempDiff = 0;
                        tempDiff = pi[k]-oldPi[k];
                        if(Math.abs(tempDiff)>diff){
                            diff = Math.abs(tempDiff);
                        }
                        oldPi[k] = pi[k];
                    }

                    if(diff<m_lbfgsConverge){
//                        System.out.print("diff\t"+diff+"finish update pi");
                        break;
                    }

                } while (iflag[0] != 0);
            } catch (LBFGS.ExceptionWithIflag e) {
                e.printStackTrace();
            }

            for(int k=0; k<fSize; k++){
                cDoc.m_sstat[k] = Math.exp(pi[k]);
//                System.out.println(cDoc.getName()+"\tcDoc.m_sstat[]"+cDoc.m_sstat[k]);
            }
        }

    }

    public double piFuncGradientVal(_ParentDoc4DCM pDoc, double gammaSum, _ChildDoc cDoc, double[]pi, double[] piGradient){
        double funcVal = 0;
        Arrays.fill(piGradient, 0);

        double expPiSum = 0;
        double[] expPi = new double[number_of_topics];
        for(int k=0; k<number_of_topics; k++){
            expPi[k] = Math.exp(pi[k]);
            expPiSum += expPi[k];
        }

        double constantGradient = 0;
        for(int k=0; k<number_of_topics; k++){
            constantGradient += (m_alpha_c[k]-expPi[k])*Utils.digamma(expPiSum);
        }

        funcVal -= Utils.lgamma(expPiSum);
        for(int k=0; k<number_of_topics; k++){
            funcVal += (m_alpha_c[k] - 1) * (Utils.digamma(expPi[k]) - Utils.digamma(expPiSum));
            funcVal -= (expPi[k]-1)*(Utils.digamma(expPi[k])-Utils.digamma(expPiSum));
            funcVal += Utils.lgamma(expPi[k]);

            piGradient[k] = (m_alpha_c[k]-expPi[k])*Utils.digamma(expPi[k]);
//            piGradient[k] -= (m_alpha_c[k]-expPi[k])*Utils.digamma(expPiSum);
        }

        _SparseFeature[] cDocFvs = cDoc.getSparse();
        for(int n=0; n<cDocFvs.length; n++) {
            int wID = cDocFvs[n].getIndex();
            double wVal = cDocFvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                funcVal += cDoc.m_phi[n][k]*wVal*(Utils.digamma(expPi[k])-Utils.digamma(expPiSum));
                funcVal -= cDoc.m_phi[n][k]*wVal*(Utils.dotProduct(expPi, pDoc.m_sstat)/(expPiSum*gammaSum*cDoc.m_zeta));

                constantGradient += cDoc.m_phi[n][k]*wVal*Utils.digamma(expPi[k])+cDoc.m_phi[n][k]*Utils.digamma(expPiSum);
//                piGradient[k] += cDoc.m_phi[n][k]*wVal*Utils.digamma(expPi[k])+cDoc.m_phi[n][k]*Utils.digamma(expPiSum);

                double temp = pDoc.m_sstat[k]*expPiSum-Utils.dotProduct(expPi, pDoc.m_sstat);

                piGradient[k] -= cDoc.m_phi[n][k]*wVal*temp/(expPiSum*expPiSum*gammaSum*cDoc.m_zeta);
            }

        }

        for(int k=0; k<number_of_topics; k++){
            piGradient[k]  -= constantGradient;
            piGradient[k] = piGradient[k]*expPi[k];
            piGradient[k] = 0-piGradient[k];
        }

        return -funcVal;
    }

    public void updateZeta4Child(_ParentDoc4DCM pDoc){
        double totalGamma4Parent = Utils.sumOfArray(pDoc.m_sstat);
        for(_ChildDoc cDoc:pDoc.m_childDocs){
            double totalPi4Child = Utils.sumOfArray(cDoc.m_sstat);
            double gammaPiInnerProd = Utils.dotProduct(pDoc.m_sstat, cDoc.m_sstat);
            cDoc.m_zeta = gammaPiInnerProd/(totalGamma4Parent*totalPi4Child);
        }
    }


    //to be improved
    double logSumOfExponentials(double[] xs){
        if(xs.length == 1){
            return xs[0];
        }

        double sum = 0;

        double max = xs[0];
        for(int i=1; i<xs.length; i++){
            if(xs[i]>max){
                max = xs[i];
            }
        }

        for (int i = 0; i < xs.length; i++) {
            if (!Double.isInfinite(xs[i])) {
                // if the gap between the value and the maximum value is too small,
                //the exponential will become zero
                if((max-xs[i])>300){
                    continue;
                }else{
                    sum += Math.exp(xs[i]-max);
                }

            }
        }

        return Math.log(sum)+max;
    }

    public void updateEta4Child(_ParentDoc4DCM pDoc){
        for(_ChildDoc cDoc:pDoc.m_childDocs) {
            _SparseFeature[] fvs = cDoc.getSparse();
            for (int n = 0; n < fvs.length; n++) {
                int wId = fvs[n].getIndex();
                double wVal = fvs[n].getValue();

                for (int k = 0; k < number_of_topics; k++) {
                    cDoc.m_phi[n][k] = Utils.digamma(pDoc.m_sstat[k])+Utils.digamma(cDoc.m_sstat[k]);
                    cDoc.m_phi[n][k] += Utils.digamma(pDoc.m_lambda_stat[k][wId])-Utils.digamma(pDoc.m_lambda_topicStat[k]);
                }

                double logSum = logSumOfExponentials(cDoc.m_phi[n]);

                if(Double.isInfinite(logSum)){
                    System.out.println("infinite");
                    System.out.println("this doc\t"+cDoc.getName()+"\t"+"this word has a total biased probability assignment\t"+m_corpus.getFeature(wId));
                }
                if(Double.isNaN(logSum)){
                    System.out.println("nan");
                    for(int k=0; k<number_of_topics; k++)
                        System.out.println("cDoc.m_phi\t"+cDoc.m_phi[n][k]);
                }
                double phiSum = 0;
                for(int k=0; k<number_of_topics; k++){
                    if((cDoc.m_phi[n][k]-logSum)<-200){
                        cDoc.m_phi[n][k] = 1e-20;
                    }else {
                        cDoc.m_phi[n][k] = Math.exp(cDoc.m_phi[n][k] - logSum);
                    }
                    phiSum += cDoc.m_phi[n][k];
                }

                if(Math.abs(phiSum-1)>1) {
                    System.out.println("phiSum for comment\t" + phiSum);
                    for(int k=0; k<number_of_topics; k++)
                        System.out.println("m_phi\t"+cDoc.m_phi[n][k]);
                }

                if(Double.isNaN(phiSum)){
                    for(int k=0; k<number_of_topics; k++){
                        System.out.println("pDoc.m_sstat[k]\t"+pDoc.m_sstat[k]);
                        System.out.println("cDoc.m_sstat[k]\t"+cDoc.m_sstat[k]);
                        System.out.println("pDoc.m_lambda_stat[k][wId]\t"+pDoc.m_lambda_stat[k][wId]);
                        System.out.println("pDoc.m_lambda_topicStat[k]\t"+pDoc.m_lambda_topicStat[k]);
                        System.out.println("cDoc.m_phi[n][k]\t"+cDoc.m_phi[n][k]);
                    }
                }
            }
        }
    }

    @Override
    public void calculate_M_step(int iter){

        if(iter%5!=4)
            return;
//        updateAlpha4Parent();
//        updateAlpha4Child();
        updateBeta();

        for(int k=0; k<number_of_topics; k++){
            System.out.println("alpha\t"+m_alpha[k]);
            System.out.println("alpha_c\t"+m_alpha_c[k]);
            for(int v=0; v<vocabulary_size; v++){
                System.out.println("beta\t"+m_beta[k][v]);
            }
        }
    }

    public void updateAlpha4Parent(){
//        double[] alphaGradient = new double[number_of_topics];
//        double[] alphaDiag = new double[number_of_topics];
//
//        Arrays.fill(alphaGradient,0);
//        Arrays.fill(alphaDiag, 0);
//
//        double alphaHessianConstant = 0;
//        double alphaSum = Utils.sumOfArray(m_alpha);
//
//        for(int k=0; k<number_of_topics; k++){
//            alphaGradient[k] = m_parentDocNum*(Utils.digamma(alphaSum)-Utils.digamma(m_alpha[k]));
//            alphaGradient[k] += m_alpha_stat[k];
//
//            alphaDiag[k] = -m_parentDocNum*(Utils.trigamma(m_alpha[k]));
//        }
//
//        alphaHessianConstant = m_parentDocNum*Utils.trigamma(alphaSum);
//
//        updateParamViaNewtonMethod(m_alpha, alphaGradient, alphaDiag, alphaHessianConstant);
        updateParamViaNewtonMethod(m_alpha, m_parentDocNum, m_alpha_stat);

    }

    public void updateAlpha4Child(){

//        double[] alphaCGradient = new double[number_of_topics];
//        double[] alphaCDiag = new double[number_of_topics];

//        Arrays.fill(alphaCGradient,0);
//        Arrays.fill(alphaCDiag, 0);
//
//        double alphaCHessianConstant = 0;
//        double alphaCSum = Utils.sumOfArray(m_alpha_c);
//
//        for(int k=0; k<number_of_topics; k++){
//            alphaCGradient[k] = m_childDocNum*(Utils.digamma(alphaCSum)-Utils.digamma(m_alpha_c[k]));
//            alphaCGradient[k] += m_alpha_c_stat[k];
//
//            alphaCDiag[k] = -m_childDocNum*(Utils.trigamma(m_alpha_c[k]));
//        }

//        alphaCHessianConstant = m_childDocNum*Utils.trigamma(alphaCSum);

        updateParamViaNewtonMethod(m_alpha_c, m_childDocNum, m_alpha_c_stat);

    }

    public void updateBeta(){
        for(int k=0; k<number_of_topics; k++){
//            double[] betaGradient = new double[vocabulary_size];
//            double[] betaDiag = new double[vocabulary_size];
//
//            Arrays.fill(betaGradient,0);
//            Arrays.fill(betaDiag, 0);
//
//            double betaHessianConstant = 0;
//            double betaSum = Utils.sumOfArray(m_beta[k]);
//
//            for(int v=0; v<vocabulary_size; v++){
//                betaGradient[v] = m_parentDocNum*(Utils.digamma(betaSum)-Utils.digamma(m_beta[k][v]));
//                betaGradient[v] += m_beta_stat[k][v];
//
//                betaDiag[v] = -m_parentDocNum*(Utils.trigamma(m_beta[k][v]));
//            }
//
//            betaHessianConstant = m_parentDocNum*Utils.trigamma(betaSum);
//
//            updateParamViaNewtonMethod(m_beta[k], betaGradient, betaDiag, betaHessianConstant);
//
            updateParamViaNewtonMethod(m_beta[k], m_parentDocNum, m_beta_stat[k]);

        }
    }

    //update parameters via newton method
    protected void updateParamViaNewtonMethod(double[] param, double paramMultiplier, double[] paramConstant){
        int iterIndex=0;
        double diff = 0;
        int paramSize = param.length;
        double[] paramUpdate = new double[param.length];
        double[] paramDiag = new double[param.length];
        double[] paramGradient = new double[param.length];

        double paramHessianConstant = 0;
        double paramSum = 0;

        do {
            paramHessianConstant = 0;
            paramSum = Utils.sumOfArray(param);
            Arrays.fill(paramUpdate, 0);
            Arrays.fill(paramDiag, 0);
            Arrays.fill(paramGradient, 0);

            for(int i=0; i<paramSize; i++){
                paramGradient[i] = paramMultiplier*(Utils.digamma(paramSum)-Utils.digamma(param[i]));
                paramGradient[i] += paramConstant[i];

                paramDiag[i] = -paramMultiplier*(Utils.trigamma(param[i]));
            }
            paramHessianConstant = paramMultiplier*Utils.trigamma(paramSum);

            double paramDiagInverseSum = 0;
            double paramDiagInverseParamGradientSum = 0;
            for (int i = 0; i < param.length; i++) {
                paramDiagInverseSum += 1 / paramDiag[i];
                paramDiagInverseParamGradientSum += paramGradient[i] / paramDiag[i];
            }
            double c = 0;
            c = (1.0 / paramHessianConstant) + paramDiagInverseSum;
            c =  paramDiagInverseParamGradientSum/c;

            for (int i = 0; i < param.length; i++) {
                paramUpdate[i] = (paramGradient[i] - c) / paramDiag[i];
                if(Math.abs(paramUpdate[i])>diff)
                    diff = Math.abs(paramUpdate[i]);
                param[i] -= paramUpdate[i];
            }


            if(diff<m_varConverge)
                break;

        }while(++iterIndex<m_varMaxIter);

    }

    @Override
    protected void finalEst(){
        for(_Doc d:m_trainSet){
            estThetaInDoc(d);
            if(d instanceof  _ParentDoc4DCM)
                estTopicWordDistribution4Parent((_ParentDoc4DCM)d);
        }

//        for(int k=0; k<number_of_topics; k++){
//            double betaSum = Utils.sumOfArray(m_beta[k]);
//            for(int v=0; v<vocabulary_size; v++){
//                m_beta[k][v] = m_beta[k][v]/betaSum;
//            }
//        }
    }

    protected void estTopicWordDistribution4Parent(_ParentDoc4DCM pDoc){
        for(int k=0; k<number_of_topics; k++){
            for(int v=0; v<vocabulary_size; v++){
                pDoc.m_wordTopic_prob[k][v] = pDoc.m_lambda_stat[k][v]/pDoc.m_lambda_topicStat[k];
            }
        }
    }

    @Override
    public double calculate_log_likelihood(_Doc d){

        _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
        double logLikelihood = 0;
        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        double alphaSum = Utils.sumOfArray(m_alpha);

        logLikelihood += Utils.lgamma(alphaSum);
        logLikelihood -= Utils.lgamma(gammaSum);
        for(int k=0; k<number_of_topics; k++){
            logLikelihood += -Utils.lgamma(m_alpha[k])+(m_alpha[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
            logLikelihood += Utils.lgamma(pDoc.m_sstat[k]);
            logLikelihood -= (pDoc.m_sstat[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
        }

        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                double updateLikelihood = 0;

                updateLikelihood -= pDoc.m_phi[n][k]*(Math.log(pDoc.m_phi[n][k]));
                if(Double.isInfinite(updateLikelihood)){
                    System.out.println("\nlikelihood\t"+updateLikelihood+"\t"+logLikelihood);
                }
                updateLikelihood += pDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));

                if(Double.isInfinite(updateLikelihood)){
                    System.out.println("\nlikelihood\t"+updateLikelihood+"\t"+logLikelihood);
                }

                updateLikelihood += pDoc.m_phi[n][k]*wVal*(Utils.digamma(pDoc.m_lambda_stat[k][wID])-Utils.digamma(pDoc.m_lambda_topicStat[k]));


                if(Double.isInfinite(updateLikelihood)){
                    System.out.println("\nlikelihood\t"+updateLikelihood+"\t"+logLikelihood);
                    System.out.println("wVal\t"+wVal);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_sstat[k]\t"+pDoc.m_sstat[k]);
                    System.out.println("gammaSum\t"+gammaSum);
                    System.out.println("pDoc.m_lambda_stat[k][wID]\t"+pDoc.m_lambda_stat[k][wID]+"\t"+Utils.digamma(pDoc.m_lambda_stat[k][wID]));
                    System.out.println("pDoc.m_lambda_topicStat[k]\t"+pDoc.m_lambda_topicStat[k]+"\t"+Utils.digamma(pDoc.m_lambda_topicStat[k]));
                }

                logLikelihood += updateLikelihood;
//                System.out.println("loglikelihood\t"+logLikelihood);

                if(Double.isNaN(updateLikelihood)){
                    System.out.println("\nlikelihood\t"+updateLikelihood+"\t"+logLikelihood);
                    System.out.println("wVal\t"+wVal);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_sstat[k]\t"+pDoc.m_sstat[k]);
                    System.out.println("gammaSum\t"+gammaSum);
                    System.out.println("pDoc.m_lambda_stat[k][wID]\t"+pDoc.m_lambda_stat[k][wID]);
                    System.out.println("pDoc.m_lambda_topicStat[k]\t"+pDoc.m_lambda_topicStat[k]);
                }


                if(Double.isInfinite(updateLikelihood)){
                    System.out.println("\nlikelihood\t"+updateLikelihood+"\t"+logLikelihood);
                    System.out.println("wVal\t"+wVal);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                    System.out.println("pDoc.m_sstat[k]\t"+pDoc.m_sstat[k]);
                    System.out.println("gammaSum\t"+gammaSum);
                    System.out.println("pDoc.m_lambda_stat[k][wID]\t"+pDoc.m_lambda_stat[k][wID]);
                    System.out.println("pDoc.m_lambda_topicStat[k]\t"+pDoc.m_lambda_topicStat[k]);
                }
            }
        }

        double alphaCSum = Utils.sumOfArray(m_alpha_c);

        for(_ChildDoc cDoc:pDoc.m_childDocs){
            logLikelihood += Utils.lgamma(alphaCSum);
            double piSum = Utils.sumOfArray(cDoc.m_sstat);

            logLikelihood -= Utils.lgamma(piSum);

            for (int k = 0; k < number_of_topics; k++) {
                logLikelihood -= Utils.lgamma(m_alpha_c[k]);
                logLikelihood += (m_alpha_c[k] - 1) * (Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum));

                logLikelihood += Utils.lgamma(cDoc.m_sstat[k]);
                logLikelihood -= (cDoc.m_sstat[k]-1)*(Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
            }

            _SparseFeature[] cDocFvs = cDoc.getSparse();
            for(int n=0; n<cDocFvs.length; n++) {
                int wID = cDocFvs[n].getIndex();
                double wVal = cDocFvs[n].getValue();
                for(int k=0; k<number_of_topics; k++){
                    logLikelihood += cDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum)+Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
                    logLikelihood -= cDoc.m_phi[n][k]*(Utils.dotProduct(cDoc.m_sstat, pDoc.m_sstat)/(piSum*gammaSum*cDoc.m_zeta)+Math.log(cDoc.m_zeta)-1);
                    logLikelihood += wVal*cDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_lambda_stat[k][wID])-Utils.digamma(pDoc.m_lambda_topicStat[k]));

                    logLikelihood -= cDoc.m_phi[n][k]*Math.log(cDoc.m_phi[n][k]);

                    if(Double.isInfinite(logLikelihood)){
                        System.out.println("\ncDoc likelihood\t"+"\t"+logLikelihood);
                        System.out.println("cDoc.m_phi[n][k]\t"+cDoc.m_phi[n][k]);
//                System.out.println("pDoc.m_phi[n][k]\t"+pDoc.m_phi[n][k]);
                        System.out.println("pDoc.m_lambda_stat[][]\t"+pDoc.m_lambda_topicStat[k]);
                        System.out.println("cDoc.m_sstat[k]\t" + cDoc.m_sstat[k]);
                        System.out.println("piSum\t" + piSum);
//                    System.out.println("pDoc.m_lambda_stat[k][wID]\t" + pDoc.m_lambda_stat[k][wID]);
//                    System.out.println("pDoc.m_lambda_topicStat[k]\t" + pDoc.m_lambda_topicStat[k]);
                        System.out.println("cDoc zeta\t"+cDoc.m_zeta);

                    }

                }
            }


        }

        for(int k=0; k<number_of_topics; k++){
            double betaSum = Utils.sumOfArray(m_beta[k]);

            logLikelihood += Utils.lgamma(betaSum);
            logLikelihood -= Utils.lgamma(pDoc.m_lambda_topicStat[k]);
            for(int v=0; v<vocabulary_size; v++) {
                logLikelihood -= Utils.lgamma(m_beta[k][v]);
                logLikelihood += (m_beta[k][v]-1)*(Utils.digamma(pDoc.m_lambda_stat[k][v])-Utils.digamma(pDoc.m_lambda_topicStat[k]));

                logLikelihood += Utils.lgamma(pDoc.m_lambda_stat[k][v]);
                logLikelihood -= (pDoc.m_lambda_stat[k][v]-1)*(Utils.digamma(pDoc.m_lambda_stat[k][v])-Utils.digamma(pDoc.m_lambda_topicStat[k]));
            }
        }

//        System.out.println("doc \t"+pDoc.getName()+"\t likelihood \t"+logLikelihood);
        return logLikelihood;
    }


    @Override
    protected double calculate_log_likelihood() {
        double corpusLogLikelihood = 0;
        for(_Doc d:m_trainSet){
            if(d instanceof _ParentDoc4DCM){
                _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
                corpusLogLikelihood += calculate_log_likelihood(pDoc);
            }
        }
        return corpusLogLikelihood;
    }


}
