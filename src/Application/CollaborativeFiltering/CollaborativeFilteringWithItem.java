package Application.CollaborativeFiltering;

import java.util.ArrayList;

import structures._Item;
import structures._User;
import utils.Utils;

public class CollaborativeFilteringWithItem extends CollaborativeFiltering{

	public CollaborativeFilteringWithItem(ArrayList<_User> users, int fs) {
		super(users, fs);
	}
	
	// calculate the ranking score for each review of each user.
	// This similarity is simply based on BoW of item and BoW of user.
	@Override
	public double calculateRankScore(_User u, String itemID){
		_Item item = m_itemMap.get(itemID);
		return Utils.cosine(u.getBoWProfile(), item.getBoWProfile());
	}

}
