CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE student (
    student_id  INTEGER PRIMARY KEY,
    ege         DOUBLE PRECISION,
    workload_ze DOUBLE PRECISION,
    has_job     INTEGER CHECK (has_job IN (0, 1)),
    gender      TEXT,
    origin      TEXT
);
CREATE TABLE course (
    course_id        INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    semester         INTEGER NOT NULL,
    form             TEXT NOT NULL CHECK (form IN ('zachet', 'difzachet', 'exam', 'coursework')),
    ze               INTEGER NOT NULL CHECK (ze > 0),
    difficulty       DOUBLE PRECISION NOT NULL,
    prereq_course_id INTEGER REFERENCES course (course_id)
);
CREATE TABLE checkpoint_schedule (
    course_id INTEGER NOT NULL REFERENCES course (course_id),
    week      INTEGER NOT NULL CHECK (week >= 1),
    PRIMARY KEY (course_id, week)
);
CREATE TABLE enrollment (
    student_id    INTEGER NOT NULL REFERENCES student (student_id),
    course_id     INTEGER NOT NULL REFERENCES course (course_id),
    dropped       INTEGER NOT NULL CHECK (dropped IN (0, 1)),
    admitted      INTEGER NOT NULL CHECK (admitted IN (0, 1)),
    passed        INTEGER NOT NULL CHECK (passed IN (0, 1)),
    grade         INTEGER CHECK (grade BETWEEN 2 AND 5),
    retake_passed INTEGER CHECK (retake_passed IN (0, 1)),
    PRIMARY KEY (student_id, course_id)
);
CREATE TABLE weekly_activity (
    student_id INTEGER NOT NULL,
    course_id  INTEGER NOT NULL,
    week       INTEGER NOT NULL CHECK (week >= 1),
    logins     DOUBLE PRECISION,
    minutes    DOUBLE PRECISION,
    PRIMARY KEY (student_id, course_id, week),
    FOREIGN KEY (student_id, course_id) REFERENCES enrollment (student_id, course_id)
);
CREATE TABLE checkpoint (
    student_id INTEGER NOT NULL,
    course_id  INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    score      DOUBLE PRECISION CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    PRIMARY KEY (student_id, course_id, week),
    FOREIGN KEY (student_id, course_id) REFERENCES enrollment (student_id, course_id),
    FOREIGN KEY (course_id, week) REFERENCES checkpoint_schedule (course_id, week)
);
CREATE TABLE audit_log (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts         TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    student_id INTEGER,
    details    TEXT
);
CREATE INDEX idx_activity_course_week ON weekly_activity (course_id, week);
CREATE INDEX idx_audit_student ON audit_log (student_id);
