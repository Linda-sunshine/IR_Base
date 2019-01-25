package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * The analyzer is used to further analyze each post in StackOverflow, i.e., question and answer
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class MultiThreadedStackOverflowAnalyzer  extends MultiThreadedNetworkAnalyzer {

    public MultiThreadedStackOverflowAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // Load one file as a user here.
    protected void loadUser(String filename, int core){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            String userID = extractUserID(file.getName()); //UserId is contained in the filename.

            // Skip the first line since it is user name.
            reader.readLine();

            String source;
            int postId = -1, parentId = -1, score = -1;
            ArrayList<_Review> reviews = new ArrayList<_Review>();
            int index = 0;
            _Review review;
            long timestamp;
            while((line = reader.readLine()) != null){
                postId = Integer.valueOf(line.trim());
                source = reader.readLine(); // review content
                parentId = Integer.valueOf(reader.readLine().trim()); // parentId
                score = Integer.valueOf(reader.readLine()); // ylabel
                timestamp = Long.valueOf(reader.readLine());

                review = new _Review(-1, postId, source, 0, parentId, userID, timestamp);
                if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
                    reviews.add(review);
                    review.setID(index++);
                }
            }
            if(reviews.size() > 1){//at least one for adaptation and one for testing
                synchronized (m_allocReviewLock) {
                    m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
                    m_corpus.addDocs(reviews);
                }
            } else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
                review = reviews.get(0);
                synchronized (m_rollbackLock) {
                    rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
                }
            }
            reader.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // one map for indexing all questions
    HashMap<Integer, _Review> m_questionMap = new HashMap<>();
    HashMap<Integer, _Review> m_answerMap = new HashMap<>();
    HashMap<Integer, ArrayList<Integer>> m_questionAnswersMap = new HashMap<>();

    // assume we already have the cv index for all the documents
    public void buildQuestionAnswerMap() {

        // step 1: find all the questions in the data first
        for (_User u : m_users) {
            for (_Review r : u.getReviews()) {
                if (r.getParentId() == -1) {
                    m_questionMap.put(r.getID(), r);
                } else{
                    m_answerMap.put(r.getID(), r);
                }
            }
        }

        // step 2: find all the answers to the corresponding questions
        for (_User u : m_users) {
            // find all the questions in the data first
            for (_Review r : u.getReviews()) {
                int questionId = r.getParentId();
                if (questionId != -1 && m_questionMap.containsKey(questionId)) {
                    if (!m_questionAnswersMap.containsKey(questionId))
                        m_questionAnswersMap.put(questionId, new ArrayList<>());
                    m_questionAnswersMap.get(questionId).add(r.getID());
                }
            }
        }

        // step 3: calculate stat of the answers to the questions
        double avg = 0;
        int lgFive = 0;
        for (int qId : m_questionAnswersMap.keySet()) {
            avg += m_questionAnswersMap.get(qId).size();
            if (m_questionAnswersMap.get(qId).size() > 5)
                lgFive++;
        }
        avg /= m_questionAnswersMap.keySet().size();
        System.out.format("[stat] Total questions: %d, questions with answers: %d,questions with >5 answers: %d, avg anser: %.2f\n",
                m_questionMap.size(), m_questionAnswersMap.size(), lgFive, avg);
    }

    // the function selects questions for candidate recommendation
    ArrayList<Integer> m_selectedQuestions = new ArrayList<>();
    ArrayList<Integer> m_selectedAnswers = new ArrayList<>();

    public void selectQuestions4Recommendation(){
        super.constructUserIDIndex();
        for(int qId: m_questionAnswersMap.keySet()){
            ArrayList<Integer> answers = m_questionAnswersMap.get(qId);
            int nuOfAns = m_questionAnswersMap.get(qId).size();
            if(nuOfAns > 1 && nuOfAns <= 5){
                boolean flag = true;
                for(int aId: answers){
                    _User user = m_users.get(m_userIDIndex.get(m_answerMap.get(aId).getUserID()));
                    if(!containsQuestion(user)){
                        flag = false;
                    }
                }
                if(flag) {
                    m_selectedQuestions.add(qId);
                    m_selectedAnswers.addAll(answers);
                }
            }
        }

        System.out.println("Total number of valid questions: " + m_selectedQuestions.size());
        System.out.println("Total number of answers: " + m_selectedAnswers.size());
    }

    // remove connections based on selected questions and answers

    HashMap<String, HashSet<String>> m_testInteractions = new HashMap<>();
    HashMap<String, HashSet<Integer>> m_testNonInteractions = new HashMap<>();

    public void refineNetwork4Recommendation(int time, String prefix){
        int remove = 0;
        // collect the testing interactions
        HashSet<Integer> removeQs = new HashSet<Integer>();
        for(int qId: m_selectedQuestions){
            String uiId = m_questionMap.get(qId).getUserID();
            HashSet<String> uiFrds = m_networkMap.get(uiId);
            if(uiFrds == null){
                System.out.println("The user does not have any friends!");
                removeQs.add(qId);
                continue;
            }
            for(int aId: m_questionAnswersMap.get(qId)){
                String ujId = m_answerMap.get(aId).getUserID();
                HashSet<String> ujFrds = m_networkMap.get(ujId);
                if(uiFrds != null && ujFrds != null && uiFrds.contains(ujId) && ujFrds.contains(uiId)){
                    if(!m_testInteractions.containsKey(uiId))
                        m_testInteractions.put(uiId, new HashSet<>());
                    if(!m_testInteractions.containsKey(ujId))
                        m_testInteractions.put(ujId, new HashSet<>());
                    m_testInteractions.get(uiId).add(ujId);
                    m_testInteractions.get(ujId).add(ujId);
                }
            }
        }
        System.out.format("%d questions of users don't have any friends!\n", removeQs.size());

        int count = 0;
        int[] indexes = new int[removeQs.size()];
        for(int q: removeQs){
            indexes[count++] = m_selectedQuestions.indexOf(q);
        }
        Arrays.sort(indexes);
        for(int i=indexes.length-1; i>=0; i--){
            m_selectedQuestions.remove(indexes[i]);
        }
        // sample the testing non-interactions
        sampleNonInteractions(time);
        saveNonInteractions(prefix, time);
        saveInteractions(prefix);

        // remove the testing interactions from the training network
        for(String ui: m_testInteractions.keySet()){
            for(String uj: m_testInteractions.get(ui)){
                m_networkMap.get(ui).remove(uj);
                m_networkMap.get(uj).remove(ui);
                remove += 2;
            }
        }
        System.out.println(remove + " edges are removed!!!");
    }

    public void sampleNonInteractions(int time) {
        HashSet<String> interactions = new HashSet<>();
        ArrayList<Integer> nonInteractions = new ArrayList<Integer>();

        for (String uid : m_testInteractions.keySet()) {

            int i = m_userIDIndex.get(uid);
            _User ui = m_users.get(i);
            interactions = m_networkMap.get(uid);
            nonInteractions.clear();

            for (int j = 0; j < m_users.size(); j++) {
                if (i == j) continue;
                if (interactions.contains(j)) continue;
                nonInteractions.add(j);
            }

            int number = time * m_testInteractions.get(uid).size();
            m_testNonInteractions.put(uid, sampleNonInteractions(nonInteractions, number));
        }

    }

    public void saveInteractions(String prefix){
        try{
            String filename = prefix + "Interactions4Recommendations_test.txt";
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String ui: m_testInteractions.keySet()){
                writer.write(ui + "\t");
                for(String uj: m_testInteractions.get(ui)){
                    writer.write(uj + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' interactions are written in %s.\n", m_testInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveNonInteractions(String prefix, int time){
        try{
            String filename = String.format("%sNonInteractions_time_%d_Recommendations.txt", prefix, time);
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_testNonInteractions.keySet()){
                writer.write(uid + "\t");
                for(int nonIdx: m_testNonInteractions.get(uid)){
                    String nonId = m_users.get(nonIdx).getUserID();
                    writer.write(nonId + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' non-interactions are written in %s.\n", m_testNonInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public boolean containsQuestion(_User u){
        for(_Review r: u.getReviews()){
            if(r.getParentId() == -1)
                return true;
        }
        return false;
    }

    // we assign cv index for all the reviews in the corpus
    public void assignCVIndex4AnswerRecommendation(){
        int unseen = 0, seen = 0;
        for(_Doc d: m_corpus.getCollection()){
            _Review r = (_Review) d;
            if(m_selectedAnswers.contains(r.getID())){
                r.setMask4CV(0);
                unseen++;
            } else{
                r.setMask4CV(1);
                seen++;
            }
        }
        System.out.format("Train doc size: %d, test doc size: %d\n", seen, unseen);
    }

    public void printSelectedQuestionIds(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int qId: m_selectedQuestions){
                writer.write(qId+"\n");
            }
            writer.close();
            System.out.format("Finish writing %d selected questions!\n", m_selectedQuestions.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "StackOverflow"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

//        int kFold = 5, k = -1;
//        int time = 2;
//
//        String orgFriendFile = String.format("%s/%s/%sFriends_org.txt", prefix, dataset, dataset);
//        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
//        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
////        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
//        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);
//        String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction_time_%d.txt", prefix, dataset, dataset, time);

        MultiThreadedStackOverflowAnalyzer analyzer = new MultiThreadedStackOverflowAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.buildQuestionAnswerMap();
        analyzer.selectQuestions4Recommendation();

        // assign cv index for training and testing documents
        String cvIndexFile = String.format("%s/%s/%sCVIndex4Recommendation.txt", prefix, dataset, dataset);
        analyzer.assignCVIndex4AnswerRecommendation();
        analyzer.saveCVIndex(cvIndexFile);

//        int time = 4;
//        // load the interaction, remove the connections built based on the selected answers
//        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
//        String friendFile4Recommendation = String.format("%s/%s/%sFriends4Recommendation.txt", prefix, dataset, dataset);
//        String questionFile = String.format("%s/%s/%sSelectedQuestions.txt", prefix, dataset, dataset);
//        String prefix4Rec = String.format("%s/%s/%s", prefix, dataset, dataset);
//        analyzer.loadInteractions(friendFile);
//        analyzer.refineNetwork4Recommendation(time, prefix4Rec);
//        analyzer.saveNetwork(friendFile4Recommendation);
//        analyzer.printSelectedQuestionIds(questionFile);
    }
}
