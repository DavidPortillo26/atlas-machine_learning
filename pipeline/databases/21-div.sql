-- Safe division function returning 0 when divisor is zero
DROP FUNCTION IF EXISTS SafeDiv;
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    END IF;
    RETURN a / b;
END//
DELIMITER ;
