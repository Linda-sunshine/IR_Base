package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

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

            _Review review;
            long timestamp;
            while((line = reader.readLine()) != null){
                postId = Integer.valueOf(line.trim());
                source = reader.readLine(); // review content
                parentId = Integer.valueOf(reader.readLine().trim()); // parentId
                score = Integer.valueOf(reader.readLine()); // ylabel
                timestamp = Long.valueOf(reader.readLine());

                review = new _Review(postId, source, 0, parentId, userID, timestamp);
                if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
                    reviews.add(review);
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
    HashMap<Integer, ArrayList<_Review>> m_questionAnswersMap = new HashMap<>();

    // assume we already have the cv index for all the documents
    public void buildQuestionAnswerMap() {

        // step 1: find all the questions in the data first
        for(_User u: m_users){
            for(_Review r: u.getReviews()){
                if(r.getParentId() == -1){
                    m_questionMap.put(r.getID(), r);
                }
            }
        }

        // step 2: find all the answers to the corresponding questions
        for(_User u: m_users){
            // find all the questions in the data first
            for(_Review r: u.getReviews()){
                int questionId = r.getParentId();
                if(questionId != -1 && m_questionMap.containsKey(questionId)){
                    if(!m_questionAnswersMap.containsKey(questionId))
                        m_questionAnswersMap.put(questionId, new ArrayList<>());
                    m_questionAnswersMap.get(questionId).add(r);
                }
            }
        }

        // step 3: calculate stat of the answers to the questions
        double avg = 0;
        int lgFive = 0;
        for(int qId: m_questionAnswersMap.keySet()){
            avg += m_questionAnswersMap.get(qId).size();
            if(m_questionAnswersMap.get(qId).size() > 5)
                lgFive++;
        }
        avg /= m_questionAnswersMap.keySet().size();
        System.out.format("[stat] Total questions: %d, questions with answers: %d,questions with >5 answers: %d, avg anser: %.2f\n",
                m_questionMap.size(), m_questionAnswersMap.size(), lgFive, avg);
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
//
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
    }
}
