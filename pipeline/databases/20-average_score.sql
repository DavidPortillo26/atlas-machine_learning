-- Stored procedure to compute and store a user's average score
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (
    IN p_user_id INT
)
BEGIN
    DECLARE v_avg DECIMAL(10,2);

    SELECT AVG(score) INTO v_avg
    FROM corrections
    WHERE user_id = p_user_id;

    UPDATE users
    SET average_score = v_avg
    WHERE id = p_user_id;
END//
DELIMITER ;
