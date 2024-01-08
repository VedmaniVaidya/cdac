import java.util.Scanner;

public class Q11 {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int num_matches = 21;
		while(num_matches!=1) {
			System.out.println("Players Turn: Pick 1, 2, 3, or 4 matchsticks.");
			int player_in = sc.nextInt();
			num_matches-=player_in;
			System.out.println("Remaining matches:"+num_matches);
			if (num_matches<=4) {
				System.out.println("Computer picked:"+(num_matches-1));
				num_matches = 1;
				System.out.println("Game over, player lost");
			}
			else {
				System.out.println("Computer picked 4 sticks");
				num_matches=num_matches-3;
				System.out.println("Remaining matches:"+num_matches);
			}
		}
	}

}
