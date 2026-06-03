"""
法眼AI - 法律知识库构建与RAG检索
支持：公开数据源构建 + 自有数据接入
"""

import os
import json
import re
import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import requests

# ============================================================
# 数据源配置
# ============================================================
# 公开数据源（示例，可替换为你的真实数据）
DATA_SOURCE_TYPE = "demo"  # "demo" | "postgresql" | "json" | "api"
DATA_SOURCE_PATH = "./legal_data"

# ============================================================
# 1. 法律数据结构
# ============================================================
@dataclass
class LegalDocument:
    """法律文档基类"""
    id: str
    type: str  # "statute" | "case" | "rule"
    title: str
    content: str
    metadata: dict = field(default_factory=dict)

@dataclass
class Statute(LegalDocument):
    """法规"""
    statute_id: str = ""
    level: str = ""  # "law" | "regulation" | "judicial_interpretation"
    effective_date: str = ""
    expires_date: str = ""
    category: str = ""  # 民法/刑法/行政法等

    def __post_init__(self):
        self.type = "statute"
        self.statute_id = self.id

@dataclass
class Case(LegalDocument):
    """案例"""
    case_id: str = ""
    case_number: str = ""  # 案号
    court: str = ""
    judgment_date: str = ""
    case_type: str = ""  # 民事/刑事/行政
    cause_of_action: str = ""  # 案由
    key_facts: str = ""  # 关键事实（结构化）
    ruling_points: str = ""  # 裁判要点（结构化）
    judgment_result: str = ""  # 判决结果类型

    def __post_init__(self):
        self.type = "case"
        self.case_id = self.id


# ============================================================
# 2. 数据获取器（可扩展）
# ============================================================
class LegalDataLoader:
    """法律数据加载器基类"""

    def load_statutes(self) -> list[Statute]:
        raise NotImplementedError

    def load_cases(self) -> list[Case]:
        raise NotImplementedError

    def load_all(self) -> list[LegalDocument]:
        docs = []
        docs.extend(self.load_statutes())
        docs.extend(self.load_cases())
        return docs


class DemoDataLoader(LegalDataLoader):
    """演示数据加载器 - 内置示例数据"""

    def load_statutes(self) -> list[Statute]:
        """内置民法典核心条款示例"""
        statutes = [
            Statute(
                id="民法典第675条",
                title="借款返还期限",
                content="借款人应当按照约定的期限返还借款。对借款期限没有约定或者约定不明确，依据本法第五百一十条的规定仍不能确定的，借款人可以随时返还；贷款人可以催告借款人在合理期限内返还。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-借款合同",
            ),
            Statute(
                id="民法典第676条",
                title="逾期利息",
                content="借款人未按照约定的期限返还借款的，应当按照约定或者国家有关规定支付逾期利息。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-借款合同",
            ),
            Statute(
                id="民法典第677条",
                title="提前还款",
                content="借款人可以在还款期限届满前向贷款人申请提前还款；贷款人同意的，可以提前返还借款。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-借款合同",
            ),
            Statute(
                id="民法典第667条",
                title="借款合同定义",
                content="借款合同是借款人向贷款人借款，到期返还借款并支付利息的合同。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-借款合同",
            ),
            Statute(
                id="民法典第668条",
                title="借款合同形式",
                content="借款合同应当采用书面形式，但是自然人之间借款另有约定的除外。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-借款合同",
            ),
            Statute(
                id="民间借贷司法解释第24条",
                title="借贷利率上限",
                content="借贷双方约定的利率超过合同成立时一年期贷款市场报价利率四倍的，视为高利转贷。超出部分的利息约定无效。",
                level="judicial_interpretation",
                effective_date="2020-08-20",
                category="民间借贷",
            ),
            Statute(
                id="民间借贷司法解释第25条",
                title="逾期利率计算",
                content="借贷双方既约定逾期利率，又约定违约金或者其他费用，出借人可以选择主张逾期利息、违约金或者其他费用，也可以一并主张，但是总计超过合同成立时一年期贷款市场报价利率四倍的部分，人民法院不予支持。",
                level="judicial_interpretation",
                effective_date="2020-08-20",
                category="民间借贷",
            ),
            Statute(
                id="民法典第188条",
                title="诉讼时效",
                content="向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
            Statute(
                id="民法典第189条",
                title="分期付款诉讼时效",
                content="当事人约定同一债务分期履行的，诉讼时效期间从最后一期履行期限届满之日起计算。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
            Statute(
                id="民法典第190条",
                title="无民事行为能力人诉讼时效",
                content="无民事行为能力人或者限制民事行为能力人对其法定代理人的请求权的诉讼时效期间，自该法定代理终止之日起计算。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
            Statute(
                id="民法典第500条",
                title="合同订立过程",
                content="当事人在订立合同过程中有其他违背诚信原则行为的，应当承担赔偿责任。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则-合同",
            ),
            Statute(
                id="民法典第577条",
                title="违约责任",
                content="当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-违约责任",
            ),
            Statute(
                id="民法典第584条",
                title="损失赔偿",
                content="当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；但是不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-违约责任",
            ),
            Statute(
                id="民法典第585条",
                title="违约金",
                content="当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-违约责任",
            ),
            Statute(
                id="民法典第586条",
                title="定金",
                content="当事人可以约定一方向对方给付定金作为债权的担保。定金合同自实际交付定金时成立。定金的数额不得超过主合同标的额的百分之二十，超过部分不产生定金效力。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-担保",
            ),
            Statute(
                id="民法典第587条",
                title="定金效力",
                content="债务人履行债务的，定金应当抵作价款或者收回。给付定金的一方不履行债务，或者履行债务不符合约定致使不能实现合同目的的，无权请求返还定金；收受定金的一方不履行债务，或者履行债务不符合约定致使不能实现合同目的的，应当双倍返还定金。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-合同编-担保",
            ),
            Statute(
                id="民法典第119条",
                title="合同约束力",
                content="依法成立的合同，对当事人具有法律约束力。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
            Statute(
                id="民法典第120条",
                title="侵权赔偿",
                content="民事权益受到侵害的，被侵权人有权请求侵权人承担侵权责任。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-侵权责任编",
            ),
            Statute(
                id="民法典第121条",
                title="无因管理",
                content="没有法定的或者约定的义务，为避免他人利益受损失而进行管理的人，有权请求受益人偿还由此支出的必要费用。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
            Statute(
                id="民法典第122条",
                title="不当得利",
                content="因他人没有法律根据，取得不当利益，受损失的人有权请求其返还不当利益。",
                level="law",
                effective_date="2021-01-01",
                category="民法典-总则",
            ),
        ]
        return statutes

    def load_cases(self) -> list[Case]:
        """内置类案示例（裁判要点摘要）"""
        cases = [
            Case(
                id="2020最高法民终1234号",
                title="民间借贷典型案例",
                case_number="2020最高法民终1234号",
                court="最高人民法院",
                judgment_date="2020-12-01",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="出借人向借款人转账30万元，约定年利率15%，借款期限届满后借款人未还款",
                ruling_points="民间借贷逾期利息的计算标准：双方约定的利率不超过合同成立时一年期LPR四倍的，应予支持；超过四倍的部分，不予支持。",
                judgment_result="支持原告部分诉讼请求",
            ),
            Case(
                id="2019沪01民终5678号",
                title="无书面合同借贷案例",
                case_number="2019沪01民终5678号",
                court="上海市第一中级人民法院",
                judgment_date="2019-09-15",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="双方未签订书面借款合同，出借人通过银行转账30万元，借款人主张是赠与而非借款",
                ruling_points="当事人之间通过转账方式形成借贷关系，虽无书面合同，但结合转账备注、聊天记录等证据，可以认定借贷关系成立。",
                judgment_result="认定借贷关系成立，判令还款",
            ),
            Case(
                id="2021京03民终9012号",
                title="高利贷认定案例",
                case_number="2021京03民终9012号",
                court="北京市第三中级人民法院",
                judgment_date="2021-06-20",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="出借人约定年利率36%，借款本金10万元，借款期限届满后借款人主张高利贷无效",
                ruling_points="借贷双方约定的利率超过合同成立时一年期LPR四倍的，超过部分无效，借款人仅需返还本金及合法利息。",
                judgment_result="超过LPR四倍部分的利息约定无效",
            ),
            Case(
                id="2018粤03民终3456号",
                title="逾期利息计算起点案例",
                case_number="2018粤03民终3456号",
                court="广东省深圳市中级人民法院",
                judgment_date="2018-11-10",
                case_type="民事",
                cause_of_action="金融借款合同纠纷",
                key_facts="借款人未按期还款，出借人主张从逾期之日起按合同约定利率计算逾期利息",
                ruling_points="借款人未按约定期限返还借款的，应当按照约定或者国家有关规定支付逾期利息。逾期利息的起算点为借款期限届满之次日。",
                judgment_result="支持逾期利息请求",
            ),
            Case(
                id="2022川01民终7890号",
                title="微信转账借贷案例",
                case_number="2022川01民终7890号",
                court="四川省成都市中级人民法院",
                judgment_date="2022-03-25",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="出借人通过微信向借款人转账5万元，无书面合同，借款人称是还款但无证据",
                ruling_points="微信转账记录可以作为借贷关系的证据，但需结合其他证据（如聊天记录、录音等）形成完整证据链。",
                judgment_result="综合证据认定借贷关系成立",
            ),
            Case(
                id="2017最高法民终567号",
                title="违约金与利息并存案例",
                case_number="2017最高法民终567号",
                court="最高人民法院",
                judgment_date="2017-08-30",
                case_type="民事",
                cause_of_action="借款合同纠纷",
                key_facts="借款合同约定逾期还款需支付违约金并按日计算利息，出借人同时主张两项",
                ruling_points="出借人可同时主张逾期利息和违约金，但总计不得超过法定上限（合同成立时LPR四倍）。",
                judgment_result="超过上限部分不予支持",
            ),
            Case(
                id="2020浙01民终2345号",
                title="砍头息认定案例",
                case_number="2020浙01民终2345号",
                court="浙江省杭州市中级人民法院",
                judgment_date="2020-07-15",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="出借人在本金中扣除利息后交付，借款人实际收到的金额少于借条载明金额",
                ruling_points="预先在本金中扣除利息的，应当将实际出借的金额认定为本金。借条载明金额与实际交付金额不一致的，以实际交付金额为准。",
                judgment_result="按实际交付金额认定本金",
            ),
            Case(
                id="2019苏01民终8765号",
                title="夫妻共同债务认定案例",
                case_number="2019苏01民终8765号",
                court="江苏省南京市中级人民法院",
                judgment_date="2019-12-20",
                case_type="民事",
                cause_of_action="民间借贷纠纷",
                key_facts="借款人借款用于家庭经营，配偶主张不知情，债权人主张夫妻共同债务",
                ruling_points="夫妻一方在婚姻关系存续期间以个人名义超出家庭日常生活需要所负的债务，不属于夫妻共同债务；但债权人能够证明该债务用于夫妻共同生活、共同生产经营的除外。",
                judgment_result="债权人未能证明用于家庭共同生活，不认定为夫妻共同债务",
            ),
        ]
        return cases


class SQLiteDataLoader(LegalDataLoader):
    """从SQLite数据库加载法律数据"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_statutes(self) -> list[Statute]:
        conn = self._get_connection()
        cursor = conn.cursor()

        statutes = []
        try:
            cursor.execute("""
                SELECT id, title, content, level, effective_date, expires_date, category
                FROM statutes
                WHERE effective_date IS NOT NULL
            """)
            rows = cursor.fetchall()

            for row in rows:
                statutes.append(Statute(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    level=row[3] or "",
                    effective_date=row[4] or "",
                    expires_date=row[5] or "",
                    category=row[6] or "",
                ))
        finally:
            conn.close()

        return statutes

    def load_cases(self) -> list[Case]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cases = []
        try:
            cursor.execute("""
                SELECT id, title, case_number, court, judgment_date,
                       case_type, cause_of_action, key_facts, ruling_points,
                       judgment_result
                FROM cases
                WHERE ruling_points IS NOT NULL
            """)
            rows = cursor.fetchall()

            for row in rows:
                cases.append(Case(
                    id=row[0],
                    title=row[1],
                    case_number=row[2] or "",
                    court=row[3] or "",
                    judgment_date=row[4] or "",
                    case_type=row[5] or "",
                    cause_of_action=row[6] or "",
                    key_facts=row[7] or "",
                    ruling_points=row[8] or "",
                    judgment_result=row[9] or "",
                ))
        finally:
            conn.close()

        return cases


class JSONFileDataLoader(LegalDataLoader):
    """从JSON文件加载法律数据"""

    def __init__(self, statutes_path: str, cases_path: str):
        self.statutes_path = statutes_path
        self.cases_path = cases_path

    def load_statutes(self) -> list[Statute]:
        statutes = []
        if os.path.exists(self.statutes_path):
            with open(self.statutes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    statutes.append(Statute(
                        id=item["id"],
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        level=item.get("level", ""),
                        effective_date=item.get("effective_date", ""),
                        expires_date=item.get("expires_date", ""),
                        category=item.get("category", ""),
                        metadata=item.get("metadata", {}),
                    ))
        return statutes

    def load_cases(self) -> list[Case]:
        cases = []
        if os.path.exists(self.cases_path):
            with open(self.cases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    cases.append(Case(
                        id=item["id"],
                        title=item.get("title", ""),
                        case_number=item.get("case_number", ""),
                        court=item.get("court", ""),
                        judgment_date=item.get("judgment_date", ""),
                        case_type=item.get("case_type", ""),
                        cause_of_action=item.get("cause_of_action", ""),
                        key_facts=item.get("key_facts", ""),
                        ruling_points=item.get("ruling_points", ""),
                        judgment_result=item.get("judgment_result", ""),
                        metadata=item.get("metadata", {}),
                    ))
        return cases


# ============================================================
# 3. 数据管理器
# ============================================================
class LegalDataManager:
    """法律数据管理器 - 统一入口"""

    def __init__(self, source_type: str = "demo", **kwargs):
        self.source_type = source_type

        if source_type == "demo":
            self.loader = DemoDataLoader()
        elif source_type == "sqlite":
            self.loader = SQLiteDataLoader(db_path=kwargs.get("db_path", "./legal.db"))
        elif source_type == "json":
            self.loader = JSONFileDataLoader(
                statutes_path=kwargs.get("statutes_path", "./statutes.json"),
                cases_path=kwargs.get("cases_path", "./cases.json"),
            )
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        self._documents: list[LegalDocument] = []
        self._statutes: list[Statute] = []
        self._cases: list[Case] = []

    def load(self):
        """加载所有数据"""
        print(f"正在从 [{self.source_type}] 加载法律数据...")
        self._statutes = self.loader.load_statutes()
        self._cases = self.loader.load_cases()
        self._documents = self._statutes + self._cases
        print(f"加载完成: 法规 {len(self._statutes)} 条, 类案 {len(self._cases)} 条")

    @property
    def statutes(self) -> list[Statute]:
        return self._statutes

    @property
    def cases(self) -> list[Case]:
        return self._cases

    @property
    def documents(self) -> list[LegalDocument]:
        return self._documents

    def to_rag_format(self) -> list[dict]:
        """转换为RAG所需的dict格式"""
        result = []
        for doc in self._documents:
            if isinstance(doc, Statute):
                content = doc.content
                if doc.category:
                    content = f"[{doc.category}] {content}"
            elif isinstance(doc, Case):
                content = f"案由：{doc.cause_of_action}\n关键事实：{doc.key_facts}\n裁判要点：{doc.ruling_points}"
            else:
                content = doc.content

            result.append({
                "id": doc.id,
                "type": doc.type,
                "title": doc.title,
                "content": content,
                "metadata": doc.metadata if isinstance(doc, LegalDocument) else {}
            })
        return result

    def save_to_json(self, output_path: str = "./legal_data_export.json"):
        """导出为JSON（可用于备份或传输）"""
        data = {
            "statutes": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "effective_date": s.effective_date,
                    "category": s.category,
                }
                for s in self._statutes
            ],
            "cases": [
                {
                    "id": c.id,
                    "title": c.title,
                    "case_number": c.case_number,
                    "court": c.court,
                    "judgment_date": c.judgment_date,
                    "case_type": c.case_type,
                    "cause_of_action": c.cause_of_action,
                    "key_facts": c.key_facts,
                    "ruling_points": c.ruling_points,
                    "judgment_result": c.judgment_result,
                }
                for c in self._cases
            ],
            "export_time": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"数据已导出至: {output_path}")


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    # 使用演示数据
    dm = LegalDataManager(source_type="demo")
    dm.load()

    # 查看数据
    print("\n" + "=" * 60)
    print(f"法规数量: {len(dm.statutes)}")
    print(f"类案数量: {len(dm.cases)}")

    print("\n--- 法规示例 ---")
    for s in dm.statutes[:3]:
        print(f"[{s.id}] {s.title}")

    print("\n--- 类案示例 ---")
    for c in dm.cases[:3]:
        print(f"[{c.id}] {c.title} | {c.cause_of_action}")

    # 导出为RAG格式
    rag_data = dm.to_rag_format()
    print(f"\nRAG数据格式: {len(rag_data)} 条")
    print(json.dumps(rag_data[0], ensure_ascii=False, indent=2))

    # 导出备份
    dm.save_to_json()
