Ez egy “student performance” adatállomány, amely középiskolás diákok tanulási szokásait, hátterét és vizsgaeredményeit tartalmazza, kifejezetten regressziós vagy osztályozási modellek tanítására (pl. vizsgaeredmény előrejelzésére). Az oszlopok demográfiai, tanulási szokás-, viselkedés- és egészségjellemzőket kombinálnak, 
ahogy sok nyilvános diák‑teljesítmény dataset is teszi.

## Általános teljesítmény és célváltozó

- `student_id`: Egyedi numerikus azonosító minden diáknak.
- `previous_scores`: Korábbi dolgozatok / vizsgák átlagpontszáma (0–100 skálán).
- `exam_score`: Az aktuális vizsga pontszáma (0–100), ez tipikus target regresszióhoz.

## Tanulási és iskolai viselkedés

- `hours_studied`: Az adott vizsgára készüléssel töltött órák száma.
- `study_hours_per_day`: Átlagos napi tanulási idő a félév során.
- `attendance_percent`: Óralátogatás százalékban (0–100).
- `homework_completion_rate`: Házi feladatok teljesítési aránya százalékban.
- `class_participation_score`: Órai aktivitás pontszám (kis egész skálán).
- `absences`: Hiányzások száma adott időszakban.

## Demográfia és családi háttér

- `age`: Diák életkora években.
- `gender`: Bináris kód (0 = lány, 1 = fiú).
- `grade_level`: Évfolyam számmal (pl. 9–12).
- `parental_education_level`: Szülők legmagasabb iskolai végzettsége kódolva (0–4 szint).
- `socio_economic_status`: Szocioökonómiai státusz kategóriakód (pl. kvartilis).
- `internet_access_home`: Bináris jelző, van‑e otthoni internet (0/1).

## Kiegészítő tanulási támogatás

- `test_preparation_course`: Elvégzett-e vizsgafelkészítő kurzust (0 = nem, 1 = igen).
- `group_study`: Tanul-e rendszeresen csoportban (0/1).
- `use_online_resources`: Használ-e online tananyagokat, platformokat (0/1).
- `previous_term_gpa`: Előző félévi tanulmányi átlag (pl. 0–4 skálán).

## Szabadidő és digitális viselkedés

- `screen_time_hours`: Napi teljes képernyőidő órában.
- `social_media_hours`: Ebből közösségi médiával töltött órák.
- `extracurricular_activities`: Részt vesz-e tanórán kívüli tevékenységekben (sport, klub, stb.) (0/1).

## Alvás, egészség, psziché

- `sleep_hours`: Átlagos napi alvásidő órában.
- `health_status`: Önbevallott általános egészségi állapot kód (pl. 1–5).
- `stress_level`: Tanulmányi stressz szintje kódolva (1–5).
- `sleep_quality`: Alvásminőség szubjektív értéke (1–5).
- `mental_health_support`: Kap-e mentális/pszichológiai támogatást (0/1).
