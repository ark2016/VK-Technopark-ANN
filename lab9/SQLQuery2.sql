-- ������� ������� ������
CREATE TABLE ������ (
    ������_id INT PRIMARY KEY IDENTITY(1,1),-- ������� ������_id �������� ��������� ������ � ����������������.
    �������� NVARCHAR(100) NOT NULL
);
GO

-- ������� ������� �����
CREATE TABLE ����� (
    player_id INT PRIMARY KEY IDENTITY(1,1),--������� player_id �������� ��������� ������ � ����������������.
    ������� NVARCHAR(100) NOT NULL,
    ��� NVARCHAR(100) NOT NULL,
    ������_id INT,
    CONSTRAINT FK_�����_������ FOREIGN KEY (������_id) REFERENCES ������(������_id)
	--������� ����, ������� ��������� �� ������� ������. ����������� ��������� ����������� ��� �������� � ����������.
    ON DELETE NO ACTION
    ON UPDATE NO ACTION
);
GO

-- ��������� ������ � ������� ������
INSERT INTO ������ (��������) VALUES ('������');
INSERT INTO ������ (��������) VALUES ('�����-���������');
GO

-- ��������� ������ � ������� �����
INSERT INTO ����� (�������, ���, ������_id) VALUES ('������', '����', 1);
INSERT INTO ����� (�������, ���, ������_id) VALUES ('������', '����', 2);
GO

-- ������� ������������� �� ������ ����� ����� ������
CREATE VIEW �����_������_View AS
SELECT
    �.player_id,
    �.�������,
    �.���,
    �.�������� AS ��������_�������,
    �.������_id
FROM ����� �
JOIN ������ � ON �.������_id = �.������_id;
GO
-- ������� ������������� �����_������_View, ������� �������� ������� �� ������ ����� � ������ � ���������� �� �� ������� ������_id.

-- �������� ������ �� �������������
SELECT * FROM �����_������_View;
GO

-- ������� �� �������
CREATE TRIGGER trg_Insert_�����_������_View
ON �����_������_View
INSTEAD OF INSERT
AS
BEGIN
    INSERT INTO ����� (�������, ���, ������_id)
    SELECT �������, ���, ������_id
    FROM inserted;
END;
GO

-- ������� �� ��������
CREATE TRIGGER trg_Delete_�����_������_View
ON �����_������_View
INSTEAD OF DELETE
AS
BEGIN
    DELETE FROM �����
    WHERE player_id IN (SELECT player_id FROM deleted);
END;
GO

-- ������� �� ����������
CREATE TRIGGER trg_Update_�����_������_View
ON �����_������_View
INSTEAD OF UPDATE
AS
BEGIN
    UPDATE �����
    SET ������� = inserted.�������,
        ��� = inserted.���,
        ������_id = inserted.������_id
    FROM inserted
    WHERE �����.player_id = inserted.player_id;
END;
GO
--===========================================================================================
-- ������� ������� ������ � �������� "������"
INSERT INTO ����� (�������, ���, ������_id) VALUES ('������', '�������', 1);
-- ������ ���������� ������

-- ������� ������ ����� �������������
INSERT INTO �����_������_View (�������, ���, ������_id) VALUES ('�������', '�����', 1);
-- ������ ������ ���� ������� ���������


-- ������� �������� ������ � ID 1
DELETE FROM ����� WHERE player_id = 1;
-- ������ ���������� ������

-- �������� ������ ����� �������������
DELETE FROM �����_������_View WHERE player_id = 2;
-- ������ ������ ���� ������� �������

-- ������� ���������� ������ � �������� "������" � ������ "����"
UPDATE ����� SET ������� = '������', ��� = '����' WHERE player_id = 3;
-- ������ ���������� ������

-- ���������� ������ ����� �������������
UPDATE �����_������_View SET ������� = '��������', ��� = '������' WHERE player_id = 3;
-- ������ ������ ���� ������� ���������


--===========================================================================================

-- �������� �������� �� �������
DROP TRIGGER trg_Insert_�����_������_View;
GO

-- �������� �������� �� ��������
DROP TRIGGER trg_Delete_�����_������_View;
GO

-- �������� �������� �� ����������
DROP TRIGGER trg_Update_�����_������_View;
GO

-- ������� ������� � �������������
DROP VIEW �����_������_View;
DROP TABLE �����;
DROP TABLE ������;
GO