package Application;


public class LinkPredPreprocess {
	
	// save the user-item pair to graphlab for model training.
//	public void saveFriendship(String dir){
//		int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
//		try{
//			PrintWriter trainWriter = new PrintWriter(new File(dir+"/train_all.csv"));
//			PrintWriter testWriter = new PrintWriter(new File(dir+"/test_all.csv"));
//			trainWriter.write("user_id,item_id,rating\n");
//			testWriter.write("user_id,item_id,rating\n");
//			for(_CFUser u: m_users){
//				trainUser++;
//				// print out the training pairs
//				for(_Review r: u.getTrainReviews()){
//					trainPair++;
//					trainWriter.write(String.format("%s,%s,%d\n", u.getUserID(), r.getItemID(), u.getItemRating(r.getItemID())+1));
//				}
//				String[] rankingItems = u.getRankingItems();
//				if(rankingItems == null)
//					continue;
//				testUser++;
//				for(String item: rankingItems){
//					testPair++;
//					testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), item, u.getItemRating(item)+1));
//				}
//			}
//			trainWriter.close();
//			testWriter.close();
//			System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", trainUser, trainPair, testUser, testPair);
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//		
//	}
	

}
