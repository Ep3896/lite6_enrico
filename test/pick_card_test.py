# Here I'll implement a test for the pick_card movement
# My goal is to create a perpetual loop that let the robot seach for the card, pick it up and then go back to the CameraSearching configuration
# This test has to keep track of failures and successes, so if the robot does not complete its loop, the test will fail.
# Once the test is failed, the robot has to get back to the CameraSearching configuration and the test has to start again.
# For each position, the robot has to make 100 movements, so the test will be repeated 100 times.
# At the end of all the movements, the test will print the number of successes and failures.
# Finally I will create a map that contains the squares that indicate each position the card has been placet at and so for each square 
# the rate of success will be determined. 