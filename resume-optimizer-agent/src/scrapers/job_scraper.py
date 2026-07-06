"""
招聘网站职位信息抓取模块

重要提示：
- 本模块仅提供通用框架，实际使用时请遵守目标网站的 robots.txt 和服务条款
- 建议优先使用官方开放平台 API
- 抓取频率请合理设置，避免对目标网站造成压力
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from datetime import datetime

from playwright.sync_api import sync_playwright, Page
from bs4 import BeautifulSoup
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import settings


@dataclass
class JobPosting:
    """职位信息数据结构"""
    title: str = ""
    company: str = ""
    salary: str = ""
    location: str = ""
    experience: str = ""
    education: str = ""
    description: str = ""
    requirements: str = ""
    skills: List[str] = field(default_factory=list)
    url: str = ""
    source: str = ""
    posted_at: Optional[datetime] = None


class BaseJobScraper:
    """职位抓取基类"""

    def __init__(self, proxy: Optional[str] = None, delay: int = None):
        self.proxy = proxy or settings.http_proxy
        self.delay = delay or settings.request_delay
        self.max_retries = settings.max_retries
        self.jobs: List[JobPosting] = []

    def fetch(self, keyword: str, location: str = "", page: int = 1) -> List[JobPosting]:
        """抓取职位列表，子类需重写"""
        raise NotImplementedError

    def _human_delay(self):
        """模拟人类操作间隔"""
        time.sleep(self.delay + random.uniform(0.5, 1.5))


class PlaywrightScraper(BaseJobScraper):
    """基于 Playwright 的通用浏览器抓取器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser = None
        self.context = None

    def _init_browser(self, headless: bool = True):
        """初始化浏览器"""
        p = sync_playwright().start()
        proxy_config = {"server": self.proxy} if self.proxy else None
        self.browser = p.chromium.launch(headless=headless, proxy=proxy_config)
        self.context = self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        return self.context.new_page()

    def fetch_page_content(self, url: str) -> str:
        """获取页面 HTML 内容"""
        page = self._init_browser()
        try:
            for attempt in range(self.max_retries):
                try:
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    self._human_delay()
                    return page.content()
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        finally:
            self.context.close()
            self.browser.close()

    def close(self):
        if self.browser:
            self.browser.close()


class BossZhipinScraper(PlaywrightScraper):
    """
    Boss直聘职位抓取器

    使用说明：
    1. Boss直聘网页版需要登录后才能搜索，首次运行会弹出浏览器窗口请手动登录
    2. 登录状态会保存到 data/boss_cookies.json，下次自动使用
    3. 如果登录状态过期，删除 cookie 文件重新登录即可
    4. 请合理控制抓取频率，避免对网站造成压力

    搜索URL格式: https://www.zhipin.com/web/geek/job?query=关键词&city=城市码&page=页码
    """

    # Boss直聘城市代码映射（常用城市）
    CITY_CODES = {
        "北京": "101010100",
        "上海": "101020100",
        "广州": "101280100",
        "深圳": "101280600",
        "杭州": "101210100",
        "成都": "101270100",
        "武汉": "101200100",
        "西安": "101110100",
        "南京": "101190100",
        "苏州": "101190400",
        "重庆": "101040100",
        "天津": "101030100",
        "长沙": "101250100",
        "厦门": "101230200",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cookie_path = Path("data/boss_cookies.json")
        self.cookie_path.parent.mkdir(parents=True, exist_ok=True)
        self.p = None  # playwright 实例

    def _get_city_code(self, location: str) -> str:
        """根据城市名获取城市代码"""
        if not location:
            return ""  # 空表示全国
        return self.CITY_CODES.get(location, "")

    def _ensure_login(self) -> dict:
        """
        确保已登录，返回 storage state（cookies + localStorage）
        如果有保存的cookie且未过期则直接返回，否则引导用户手动登录
        """
        if self.cookie_path.exists():
            try:
                import json
                state = json.loads(self.cookie_path.read_text(encoding="utf-8"))
                # 简单检查cookie是否过期（boss的token一般30天）
                return state
            except Exception:
                pass

        logger.warning("Boss直聘需要登录，正在打开浏览器窗口，请在 60 秒内完成登录...")
        return self._manual_login()

    def _manual_login(self) -> dict:
        """打开有头浏览器让用户手动登录，然后保存状态"""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # 使用更真实的浏览器配置
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )

            # 隐藏自动化标志
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page = context.new_page()

            # 先访问首页，再跳转到登录页
            logger.info("正在打开 Boss直聘...")
            page.goto("https://www.zhipin.com/", wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)  # 等待页面加载

            # 跳转到登录页
            page.goto("https://login.zhipin.com/", wait_until="domcontentloaded", timeout=30000)

            print("\n" + "=" * 60)
            print("  请在弹出的浏览器窗口中登录 Boss直聘")
            print("  登录成功后（看到首页），在此按回车键继续...")
            print("=" * 60 + "\n")
            input("按回车键确认已登录...")

            # 保存登录状态
            state = context.storage_state()
            self.cookie_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

            context.close()
            browser.close()

            logger.success("登录状态已保存")
            return state

    def fetch(self, keyword: str, location: str = "", page: int = 1, max_detail_jobs: int = 5) -> List[JobPosting]:
        """
        抓取 Boss直聘 职位列表，并补充详情页信息

        :param max_detail_jobs: 最多访问详情页的数量（防止请求过多触发反爬）
        """
        from playwright.sync_api import sync_playwright

        state = self._ensure_login()
        city_code = self._get_city_code(location)

        search_url = f"https://www.zhipin.com/web/geek/job?query={keyword}&page={page}"
        if city_code:
            search_url += f"&city={city_code}"

        jobs = []

        with sync_playwright() as p:
            proxy_config = {"server": self.proxy} if self.proxy else None
            browser = p.chromium.launch(
                headless=True,
                proxy=proxy_config,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )
            context = browser.new_context(
                storage_state=state,
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )

            # 隐藏自动化标志
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page_obj = context.new_page()

            try:
                logger.info(f"正在抓取列表页: {search_url}")
                page_obj.goto(search_url, wait_until="networkidle", timeout=60000)
                self._human_delay()

                # 等待职位列表加载
                page_obj.wait_for_selector(".job-card-wrapper, .search-job-result", timeout=30000)

                # 获取页面内容并解析列表
                html = page_obj.content()
                jobs = self._parse_list(html)

                if not jobs:
                    logger.warning("未解析到职位，可能页面结构已变更或需要重新登录")
                else:
                    logger.info(f"列表页解析完成，共 {len(jobs)} 个职位")

                    # 对前 N 个职位访问详情页，补充完整描述
                    jobs_to_enrich = jobs[:max_detail_jobs]
                    logger.info(f"正在补充前 {len(jobs_to_enrich)} 个职位的详情页信息...")

                    for idx, job in enumerate(jobs_to_enrich):
                        if job.url:
                            try:
                                self._enrich_job_detail(page_obj, job)
                                logger.info(f"[{idx + 1}/{len(jobs_to_enrich)}] 已补充详情: {job.title}")
                            except Exception as e:
                                logger.warning(f"抓取详情页失败 [{job.title}]: {e}")
                            self._human_delay()

            except Exception as e:
                logger.error(f"抓取失败: {e}")
                if self.cookie_path.exists():
                    self.cookie_path.unlink()
                    logger.info("已清除登录状态，下次将重新引导登录")

            finally:
                context.close()
                browser.close()

        return jobs

    def _enrich_job_detail(self, page_obj, job: JobPosting):
        """
        访问职位详情页，补充职位描述和任职要求
        在同一个浏览器 page 中完成，复用登录状态
        """
        page_obj.goto(job.url, wait_until="networkidle", timeout=30000)
        self._human_delay()

        # 等待详情页主要内容加载
        page_obj.wait_for_selector(".job-sec-text, .job-description, .detail-content", timeout=15000)

        html = page_obj.content()
        soup = BeautifulSoup(html, "html.parser")

        # 提取职位描述（通常在 .job-sec-text 中）
        desc_el = (
            soup.select_one(".job-sec-text") or
            soup.select_one(".job-description") or
            soup.select_one(".detail-content") or
            soup.select_one("[class*='description']")
        )
        if desc_el:
            job.description = desc_el.get_text(separator="\n", strip=True)

        # 提取岗位职责（可能和描述在一起，尝试拆分）
        # Boss直聘通常把"岗位职责"和"任职要求"放在同一个文本块中
        if job.description:
            desc_lower = job.description.lower()
            # 尝试按常见标题拆分
            split_markers = ["任职要求：", "任职资格：", "岗位要求：", "必备技能："]
            for marker in split_markers:
                if marker in desc_lower:
                    parts = job.description.split(marker, 1)
                    if len(parts) == 2:
                        job.description = parts[0].strip()
                        job.requirements = marker + parts[1].strip()
                        break

        # 如果没有拆分出来，把全部内容同时作为 description 和 requirements
        if not job.requirements and job.description:
            job.requirements = job.description

        # 提取技能标签（详情页可能有更完整的标签）
        skill_els = soup.select(".job-tags .tag, .job-requirements span, .skill-tag")
        if skill_els:
            detail_skills = [s.get_text(strip=True) for s in skill_els if s.get_text(strip=True)]
            # 合并去重
            existing = set(job.skills)
            for s in detail_skills:
                if s not in existing and len(s) < 20:  # 过滤过长的非技能文本
                    job.skills.append(s)

    def _parse_list(self, html: str) -> List[JobPosting]:
        """解析 Boss直聘 职位列表页"""
        soup = BeautifulSoup(html, "html.parser")
        jobs = []

        # Boss直聘职位卡片选择器（基于常见结构，可能需要根据实际页面微调）
        # 优先尝试多种可能的选择器
        items = (
            soup.select(".job-card-wrapper") or
            soup.select("[ka='search_list_1']") or
            soup.select(".job-list-box .job-card-body") or
            soup.select("ul.job-list-box > li")
        )

        logger.info(f"找到 {len(items)} 个职位卡片")

        for item in items:
            try:
                job = self._extract_job_card(item)
                if job.title:
                    jobs.append(job)
            except Exception as e:
                logger.debug(f"解析单个职位失败: {e}")
                continue

        return jobs

    def _extract_job_card(self, item) -> JobPosting:
        """从单个职位卡片提取数据"""
        job = JobPosting(source="boss")

        # 职位名称
        title_el = (
            item.select_one(".job-name") or
            item.select_one(".job-title") or
            item.select_one("a[title]")
        )
        if title_el:
            job.title = title_el.get_text(strip=True)
            job.url = title_el.get("href", "")
            if job.url and not job.url.startswith("http"):
                job.url = "https://www.zhipin.com" + job.url

        # 薪资
        salary_el = item.select_one(".salary") or item.select_one(".job-salary")
        if salary_el:
            job.salary = salary_el.get_text(strip=True)

        # 工作地点
        location_el = item.select_one(".job-area") or item.select_one(".job-location")
        if location_el:
            job.location = location_el.get_text(strip=True)

        # 公司名称
        company_el = (
            item.select_one(".company-name") or
            item.select_one(".name") or
            item.select_one("[ka='search_list_company_1']")
        )
        if company_el:
            job.company = company_el.get_text(strip=True)

        # 经验要求 & 学历（通常在标签列表中）
        tag_els = item.select(".tag-list li, .job-labels li, .info-desc")
        for tag in tag_els:
            text = tag.get_text(strip=True)
            if "经验" in text or "年" in text:
                job.experience = text
            elif text in ["本科", "大专", "硕士", "博士", "学历不限"]:
                job.education = text
            elif not job.location and ("区" in text or "路" in text):
                job.location = text

        # 技能标签
        skill_els = item.select(".skill-tags li, .tag-list li")
        job.skills = [s.get_text(strip=True) for s in skill_els if s.get_text(strip=True)]

        # 如果卡片里没有详细描述，尝试从职位链接获取（可选）
        if not job.description and job.url:
            job.description = f"详情见: {job.url}"

        return job


class MockJobScraper(BaseJobScraper):
    """
    模拟数据抓取器（用于开发和测试，无需真实抓取）
    """

    def fetch(self, keyword: str, location: str = "", page: int = 1, max_detail_jobs: int = 0) -> List[JobPosting]:
        """返回模拟的职位数据"""
        mock_jobs = [
            JobPosting(
                title=f"高级{keyword}工程师",
                company="示例科技有限公司",
                salary="25-40K·14薪",
                location=location or "上海",
                experience="3-5年",
                education="本科",
                description="负责核心业务系统的设计与开发，参与技术方案评审。",
                requirements="精通Python，熟悉Django/Flask框架；有大型分布式系统经验；熟悉MySQL、Redis、Kafka。",
                skills=["Python", "Django", "MySQL", "Redis", "Kafka", "分布式系统"],
                source="mock",
            ),
            JobPosting(
                title=f"{keyword}开发专家",
                company="某互联网大厂",
                salary="40-70K·16薪",
                location=location or "北京",
                experience="5-10年",
                education="本科",
                description="负责公司级基础架构建设，带领团队完成技术攻坚。",
                requirements="精通Python/Go至少一门语言；熟悉Kubernetes、Docker；有团队管理经验；具备良好的沟通能力。",
                skills=["Python", "Go", "Kubernetes", "Docker", "团队管理", "架构设计"],
                source="mock",
            ),
            JobPosting(
                title=f"初级{keyword}开发",
                company="创业公司A",
                salary="12-20K",
                location=location or "深圳",
                experience="1-3年",
                education="本科",
                description="参与产品功能开发，编写单元测试，修复线上问题。",
                requirements="熟悉Python基础语法；了解Web开发；有学习热情；具备良好的代码习惯。",
                skills=["Python", "Web开发", "Git", "Linux"],
                source="mock",
            ),
        ]
        return mock_jobs


class ZhaopinScraper(PlaywrightScraper):
    """
    智联招聘职位抓取器

    搜索URL格式: https://sou.zhaopin.com/?kw=关键词&city=城市代码&p=页码
    职位详情URL: https://www.zhaopin.com/jobdetail/xxx.htm

    注意：
    - 智联招聘反爬相对温和，但仍需控制抓取频率
    - 建议设置合理的 REQUEST_DELAY（默认2秒）
    """

    # 智联招聘城市代码映射
    CITY_CODES = {
        "北京": "530",
        "上海": "538",
        "广州": "763",
        "深圳": "765",
        "杭州": "653",
        "成都": "801",
        "武汉": "736",
        "西安": "854",
        "南京": "635",
        "苏州": "639",
        "重庆": "551",
        "天津": "531",
        "长沙": "749",
        "厦门": "661",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_city_code(self, location: str) -> str:
        """根据城市名获取城市代码"""
        if not location:
            return ""
        return self.CITY_CODES.get(location, "")

    def fetch(self, keyword: str, location: str = "", min_salary: int = 0, industry: str = "", page: int = 1, max_jobs: int = 10) -> List[JobPosting]:
        """
        抓取智联招聘职位列表
        :param max_jobs: 抓取职位数量（全部访问详情页），上限 20
        """
        from playwright.sync_api import sync_playwright

        city_code = self._get_city_code(location)
        max_jobs = min(max_jobs, 20)

        jobs = []

        with sync_playwright() as p:
            proxy_config = {"server": self.proxy} if self.proxy else None
            browser = p.chromium.launch(
                headless=True,
                proxy=proxy_config,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )

            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page_obj = context.new_page()

            try:
                # 分页抓取，直到收集到 max_jobs 个职位
                current_page = page
                while len(jobs) < max_jobs:
                    if city_code:
                        search_url = f"https://www.zhaopin.com/sou/jl{city_code}/kw{keyword}/p{current_page}"
                    else:
                        search_url = f"https://www.zhaopin.com/sou/kw{keyword}/p{current_page}"

                    logger.info(f"正在抓取智联招聘第 {current_page} 页: {search_url}")
                    page_obj.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                    self._human_delay()
                    page_obj.wait_for_timeout(3000)
                    html = page_obj.content()
                    page_jobs = self._parse_list(html)

                    if not page_jobs:
                        if current_page == page:
                            logger.warning("未解析到职位，可能页面结构已变更")
                        break

                    jobs.extend(page_jobs)
                    logger.info(f"第 {current_page} 页获取 {len(page_jobs)} 个，累计 {len(jobs)}/{max_jobs}")

                    if len(page_jobs) < 20:
                        break  # 最后一页通常不满

                    current_page += 1

                # 截断到 max_jobs
                if len(jobs) > max_jobs:
                    jobs = jobs[:max_jobs]

                if not jobs:
                    logger.warning("未解析到职位，可能页面结构已变更")
                else:
                    logger.info(f"列表页解析完成，共 {len(jobs)} 个职位")

                    # 二次过滤：按地点筛选（智联招聘的 city 参数有时不严格）
                    if location:
                        before_count = len(jobs)
                        filtered_jobs = self._filter_by_location(jobs, location)
                        logger.info(f"地点过滤结果: {before_count} → {len(filtered_jobs)} 个职位 (filtered is jobs: {filtered_jobs is jobs})")
                        if len(filtered_jobs) < before_count:
                            logger.info(f"按地点「{location}」过滤: {before_count} → {len(filtered_jobs)} 个职位")
                            jobs = filtered_jobs
                        elif len(filtered_jobs) == before_count:
                            # 检查是否真的全部匹配
                            matched = sum(1 for j in jobs if location in (j.location or "") or (j.location or "") in location)
                            if matched > 0:
                                logger.info(f"地点「{location}」匹配全部 {before_count} 个职位，无需过滤")
                                jobs = filtered_jobs
                            else:
                                logger.warning(f"地点过滤未生效（{before_count}个职位均不匹配「{location}」），保留全部结果")

                    # 二次过滤：按薪资筛选
                    if min_salary > 0:
                        before_count = len(jobs)
                        filtered_jobs = self._filter_by_salary(jobs, min_salary)
                        if len(filtered_jobs) < before_count:
                            logger.info(f"按最低薪资「{min_salary}元/月」过滤: {before_count} → {len(filtered_jobs)} 个职位")
                            jobs = filtered_jobs
                        else:
                            logger.info(f"薪资过滤未生效（所有职位均满足 {min_salary}元/月 或未提供薪资信息），保留全部结果")

                    # 二次过滤：按行业筛选
                    if industry:
                        before_count = len(jobs)
                        filtered_jobs = self._filter_by_industry(jobs, industry)
                        if len(filtered_jobs) < before_count:
                            logger.info(f"按行业「{industry}」过滤: {before_count} → {len(filtered_jobs)} 个职位")
                            jobs = filtered_jobs
                        else:
                            logger.info(f"行业过滤未生效（所有职位均匹配「{industry}」或未提供行业信息），保留全部结果")

                    # 对所有职位访问详情页补充完整信息
                    jobs_to_enrich = jobs
                    if jobs_to_enrich:
                        logger.info(f"正在补充 {len(jobs_to_enrich)} 个职位的详情页信息...")
                        for idx, job in enumerate(jobs_to_enrich):
                            if job.url:
                                try:
                                    self._enrich_job_detail(page_obj, job)
                                    logger.info(f"[{idx + 1}/{len(jobs_to_enrich)}] 已补充详情: {job.title}")
                                except Exception as e:
                                    logger.warning(f"抓取详情页失败 [{job.title}]: {e}")
                                self._human_delay()

            except Exception as e:
                logger.error(f"抓取失败: {e}")

            finally:
                context.close()
                browser.close()

        return jobs

    @staticmethod
    def _filter_by_location(jobs: List[JobPosting], location: str) -> List[JobPosting]:
        """
        按地点过滤职位。
        匹配规则：职位的 location 字段包含目标城市名，或目标城市名包含在 location 中。
        如果过滤后结果为空，则返回原始列表（避免全部不匹配时丢失所有数据）。
        """
        filtered = []
        for job in jobs:
            loc = job.location or ""
            if location in loc or loc in location:
                filtered.append(job)
        return filtered if filtered else jobs

    @staticmethod
    def _parse_salary(salary_str: str) -> tuple:
        """
        解析薪资字符串，返回 (min_salary, max_salary) 元组（单位：元/月）。
        支持格式：
        - "8000-12000" → (8000, 12000)
        - "8K-12K" → (8000, 12000)
        - "1.5-2万" → (15000, 20000)
        - "15-20K" → (15000, 20000)
        - "面议" 或无法解析 → (0, 0)
        """
        import re
        if not salary_str or salary_str.strip() in ("面议", "negotiable", ""):
            return (0, 0)

        salary_str = salary_str.strip().upper()

        # 匹配 "X-Y万" 格式
        match = re.search(r'([\d.]+)-([\d.]+)万', salary_str)
        if match:
            min_val = float(match.group(1)) * 10000
            max_val = float(match.group(2)) * 10000
            return (int(min_val), int(max_val))

        # 匹配 "X-YK" 或 "X-Yk" 格式
        match = re.search(r'([\d.]+)-([\d.]+)[Kk]', salary_str)
        if match:
            min_val = float(match.group(1)) * 1000
            max_val = float(match.group(2)) * 1000
            return (int(min_val), int(max_val))

        # 匹配纯数字 "X-Y" 格式
        match = re.search(r'(\d+)-(\d+)', salary_str)
        if match:
            min_val = int(match.group(1))
            max_val = int(match.group(2))
            # 如果数字较小，可能是以千为单位
            if min_val < 100:
                min_val *= 1000
                max_val *= 1000
            return (min_val, max_val)

        return (0, 0)

    @staticmethod
    def _filter_by_salary(jobs: List[JobPosting], min_salary: int) -> List[JobPosting]:
        """
        按最低薪资过滤职位。
        匹配规则：职位的薪资上限 >= 要求的最低薪资。
        没有薪资信息的职位单独统计，不直接放行。
        """
        matched = []
        no_salary_info = []
        for job in jobs:
            salary_min, salary_max = ZhaopinScraper._parse_salary(job.salary)
            if salary_max == 0:
                no_salary_info.append(job)
            elif salary_max >= min_salary:
                matched.append(job)

        # 如果有明确匹配的职位，只返回这些（不含无薪资信息的）
        if matched:
            logger.info(f"薪资过滤: {len(matched)} 个职位满足 >={min_salary}元/月, {len(no_salary_info)} 个无薪资信息被排除")
            return matched
        # 如果没有任何职位有薪资信息，返回全部并提示
        logger.warning(f"薪资过滤: 所有 {len(jobs)} 个职位均未提供薪资信息，无法过滤")
        return jobs

    @staticmethod
    def _filter_by_industry(jobs: List[JobPosting], industry: str) -> List[JobPosting]:
        """
        按行业过滤职位。
        匹配规则：职位的标题、描述、公司名称、技能标签中包含行业关键词。
        """
        filtered = []
        for job in jobs:
            # 在标题、描述、公司名称、技能中搜索行业关键词
            search_text = f"{job.title} {job.description} {job.company} {' '.join(job.skills)}".lower()
            if industry.lower() in search_text:
                filtered.append(job)

        if filtered:
            logger.info(f"行业过滤: {len(filtered)}/{len(jobs)} 个职位匹配「{industry}」")
            return filtered
        logger.warning(f"行业过滤: 没有职位匹配「{industry}」，保留全部 {len(jobs)} 个职位")
        return jobs

    def _parse_list(self, html: str) -> List[JobPosting]:
        """解析智联招聘职位列表页"""
        soup = BeautifulSoup(html, "html.parser")
        jobs = []

        # 智联招聘职位卡片选择器
        items = (
            soup.select(".joblist-box__item") or
            soup.select(".positionlist .position-item") or
            soup.select("[class*='jobitem']") or
            soup.select(".job-list li")
        )

        logger.info(f"找到 {len(items)} 个职位卡片")

        for item in items:
            try:
                job = self._extract_job_card(item)
                if job.title:
                    jobs.append(job)
            except Exception as e:
                logger.debug(f"解析单个职位失败: {e}")
                continue

        return jobs

    def _extract_job_card(self, item) -> JobPosting:
        """从单个职位卡片提取数据"""
        job = JobPosting(source="zhaopin")

        # 职位名称和链接（智联招聘实际DOM: .jobinfo__name > a）
        title_el = (
            item.select_one(".jobinfo__name") or
            item.select_one(".iteminfo__line1__jobname a") or
            item.select_one(".job-name a") or
            item.select_one("a[title]")
        )
        if title_el:
            job.title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            if href:
                job.url = href if href.startswith("http") else "https://www.zhaopin.com" + href

        # 公司名称（智联招聘实际DOM: .companyinfo__name）
        company_el = (
            item.select_one(".companyinfo__name") or
            item.select_one(".iteminfo__line1__companyname a") or
            item.select_one(".company-name a")
        )
        if company_el:
            job.company = company_el.get_text(strip=True)

        # 薪资（智联招聘实际DOM: .jobinfo__salary）
        salary_el = (
            item.select_one(".jobinfo__salary") or
            item.select_one(".iteminfo__line2__jobdesc__salary") or
            item.select_one(".job-salary")
        )
        if salary_el:
            job.salary = salary_el.get_text(strip=True)

        # 工作地点（智联招聘实际DOM: .jobinfo__other-info-item > span）
        location_el = (
            item.select_one(".jobinfo__other-info-item span") or
            item.select_one(".iteminfo__line2__jobdesc__city") or
            item.select_one(".job-area")
        )
        if location_el:
            job.location = location_el.get_text(strip=True)

        # 经验和学历（智联招聘实际DOM: .jobinfo__other-info-item 的纯文本）
        desc_els = item.select(".jobinfo__other-info-item")
        for el in desc_els:
            text = el.get_text(strip=True)
            if "经验" in text or "年" in text:
                job.experience = text
            elif text in ["本科", "大专", "硕士", "博士", "学历不限", "初中"]:
                job.education = text

        # 技能/关键词标签（只取jobinfo区域的标签，排除companyinfo区域的公司标签）
        tag_els = item.select(".jobinfo__tag .joblist-box__item-tag")
        job.skills = [s.get_text(strip=True) for s in tag_els if s.get_text(strip=True)]

        return job

    def _enrich_job_detail(self, page_obj, job: JobPosting):
        """访问职位详情页，补充完整描述"""
        page_obj.goto(job.url, wait_until="domcontentloaded", timeout=30000)
        self._human_delay()
        page_obj.wait_for_timeout(2000)

        html = page_obj.content()
        soup = BeautifulSoup(html, "html.parser")

        # 提取职位描述
        desc_el = (
            soup.select_one(".describtion__detail-content") or
            soup.select_one(".job-detail-section__detail-content") or
            soup.select_one(".position-detail") or
            soup.select_one("[class*='description']")
        )
        if desc_el:
            job.description = desc_el.get_text(separator="\n", strip=True)

        # 拆分岗位职责和任职要求
        if job.description:
            desc_lower = job.description.lower()
            split_markers = ["任职要求：", "任职资格：", "岗位要求：", "必备技能：", "职位要求："]
            for marker in split_markers:
                if marker in desc_lower:
                    parts = job.description.split(marker, 1)
                    if len(parts) == 2:
                        job.description = parts[0].strip()
                        job.requirements = marker + parts[1].strip()
                        break

        if not job.requirements and job.description:
            job.requirements = job.description

        # 补充技能标签
        skill_els = soup.select(".job-tags .tag, .skill-tags span, [class*='tag']")
        if skill_els:
            detail_skills = [s.get_text(strip=True) for s in skill_els if s.get_text(strip=True)]
            existing = set(job.skills)
            for s in detail_skills:
                if s not in existing and len(s) < 20:
                    job.skills.append(s)


class URLJobScraper(PlaywrightScraper):
    """
    通用 URL 职位抓取器

    从用户提供的任意 URL 抓取职位信息。
    支持主流招聘网站（智联、Boss、前程无忧、拉勾等）的详情页，
    同时对未知站点使用通用提取策略。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(self, urls: list = None, **kwargs) -> List[JobPosting]:
        """
        从一组 URL 抓取职位信息
        :param urls: 职位详情页 URL 列表
        """
        from playwright.sync_api import sync_playwright

        if not urls:
            logger.warning("未提供任何 URL")
            return []

        if isinstance(urls, str):
            urls = [urls]

        jobs = []

        with sync_playwright() as p:
            proxy_config = {"server": self.proxy} if self.proxy else None
            browser = p.chromium.launch(
                headless=True,
                proxy=proxy_config,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page_obj = context.new_page()

            for idx, url in enumerate(urls):
                url = url.strip()
                if not url:
                    continue
                if not url.startswith("http"):
                    url = "https://" + url

                try:
                    logger.info(f"[{idx + 1}/{len(urls)}] 正在抓取: {url}")
                    job = self._scrape_job_url(page_obj, url)
                    if job and job.title:
                        jobs.append(job)
                        logger.info(f"  ✓ 抓取成功: {job.title} @ {job.company}")
                    else:
                        logger.warning(f"  ✗ 未能提取职位信息: {url}")
                except Exception as e:
                    logger.error(f"  ✗ 抓取失败 [{url}]: {e}")

                if idx < len(urls) - 1:
                    self._human_delay()

            context.close()
            browser.close()

        logger.info(f"URL 抓取完成: 成功 {len(jobs)}/{len(urls)} 个职位")
        return jobs

    def _scrape_job_url(self, page_obj, url: str) -> JobPosting:
        """从单个 URL 提取职位信息"""
        page_obj.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._human_delay()

        # 等待页面 JS 渲染完成：优先等 h1（职位名），最多等 5 秒
        try:
            page_obj.wait_for_selector("h1", timeout=5000)
        except Exception:
            page_obj.wait_for_timeout(3000)

        html = page_obj.content()
        soup = BeautifulSoup(html, "html.parser")
        job = JobPosting(source="url", url=url)

        # ── 首页重定向检测 ──
        current_url = page_obj.url
        page_text = soup.get_text()
        homepage_signals = ["页面不小心走丢", "页面不存在", "更懂你的价值"]
        is_homepage = any(s in page_text for s in homepage_signals) or \
                      "zhaopin.com" in current_url and "jobdetail" not in current_url and "companydetail" not in current_url
        if is_homepage:
            logger.warning(f"URL 已重定向到首页或页面失效: {url} → {current_url}")
            job.title = f"[已失效] {url}"
            return job

        # ── 从 <title> 标签预提取信息（作为 fallback）──
        # 智联格式: "职位名招聘_公司名招聘 - 智联招聘"
        title_from_tag = ""
        company_from_title = ""
        if soup.title:
            raw_title = soup.title.get_text(strip=True)
            # 去除网站后缀
            for suffix in [" - 智联招聘", " - BOSS直聘", " - 前程无忧", " - 拉勾"]:
                if suffix in raw_title:
                    raw_title = raw_title.split(suffix)[0]
                    break
            if "_" in raw_title:
                parts = raw_title.split("_")
                # 职位名：去掉末尾的"招聘"
                job_raw = parts[0].strip()
                if job_raw.endswith("招聘"):
                    job_raw = job_raw[:-2]
                title_from_tag = job_raw
                # 公司名：去掉末尾的"招聘"
                if len(parts) >= 2:
                    company_raw = parts[1].strip()
                    if company_raw.endswith("招聘"):
                        company_raw = company_raw[:-2]
                    company_from_title = company_raw
            else:
                title_from_tag = raw_title.strip()

        # ── 职位名称 ──
        # 策略 1：h1 标签（智联招聘新版 DOM: h1.summary-planes__title）
        title_el = soup.select_one("h1")
        if title_el and title_el.get_text(strip=True):
            h1_text = title_el.get_text(strip=True)
            # 过滤掉明显不是职位名的 h1（如"职位描述"、"公司信息"等）
            skip_h1 = {"职位描述", "工作地点", "公司信息", "相似职位", "招聘信息", "工商信息"}
            if h1_text not in skip_h1:
                job.title = h1_text
        # 策略 2：<title> 标签解析
        if not job.title and title_from_tag:
            job.title = title_from_tag

        if not job.title:
            logger.warning(f"无法提取职位名称: {url}")
            job.title = f"[未知职位] {url}"
            return job

        # ── 公司名称 ──
        # 策略 1：公司详情链接 a[href*='company']（智联招聘最可靠的选择器）
        for sel in [
            "a[href*='companydetail']",
            "a[href*='company']",
            ".company-info a", ".detail-company a", ".job-company a",
            ".company-info .name", ".company-box .name",
            ".company-name", ".companyinfo__name",
            "[class*='company-name']", "[class*='company_name']",
        ]:
            els = soup.select(sel)
            for el in els:
                text = el.get_text(strip=True)
                # 过滤掉非公司名的链接文本（如"1 个在招职位"、"查看全部信息"等）
                if not text or len(text) < 2:
                    continue
                skip_keywords = ["在招职位", "查看全部", "查看更多", "公司介绍", "招聘信息", "工商信息"]
                if any(kw in text for kw in skip_keywords):
                    continue
                if "公司" in text or "有限" in text or len(text) >= 4:
                    job.company = text
                    break
            if job.company:
                break

        # 策略 2：<title> 标签解析的公司名
        if not job.company and company_from_title:
            job.company = company_from_title

        # ── 薪资 ──
        for sel in [
            ".salary", ".job-salary", ".jobinfo__salary",
            "[class*='salary']", "[class*='pay']",
        ]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                job.salary = el.get_text(strip=True)
                break

        # ── 地点 ──
        for sel in [
            ".job-area", ".job-location", ".jobinfo__other-info-item",
            "[class*='location']", "[class*='job-area']",
        ]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                text = el.get_text(strip=True)
                if any(c in text for c in ["市", "区", "省", "北京", "上海", "广州", "深圳", "杭州", "成都"]):
                    job.location = text
                    break

        # ── 职位描述 ──
        for sel in [
            ".job-detail", ".job-description", ".position-detail",
            ".describtion__detail-content", ".job-sec-text",
            ".detail-content", ".jobdesc",
            "[class*='job-detail']", "[class*='description']",
        ]:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 30:
                job.description = el.get_text(separator="\n", strip=True)
                break

        if not job.description:
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    job.description = text[:3000]

        # ── 拆分描述与任职要求 ──
        if job.description:
            desc_lower = job.description.lower()
            for marker in ["任职要求", "任职资格", "岗位要求", "职位要求", "必备技能", "职位要求"]:
                if marker in desc_lower:
                    parts = job.description.split(marker, 1)
                    if len(parts) == 2:
                        job.description = parts[0].strip()
                        job.requirements = marker + parts[1].strip()
                        break
        if not job.requirements and job.description:
            job.requirements = job.description

        # ── 技能标签 ──
        for sel in [
            ".job-tags .tag", ".skill-tags span", ".jobinfo__tag .joblist-box__item-tag",
            "[class*='skill-tag']", "[class*='tag']",
        ]:
            tag_els = soup.select(sel)
            if tag_els:
                skills = [s.get_text(strip=True) for s in tag_els if s.get_text(strip=True) and len(s.get_text(strip=True)) < 20]
                if skills:
                    job.skills = skills
                    break

        # ── 经验 & 学历 ──
        for sel in ["[class*='experience']", "[class*='work-year']"]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                job.experience = el.get_text(strip=True)
                break

        for sel in ["[class*='education']", "[class*='degree']"]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                job.education = el.get_text(strip=True)
                break

        return job


def get_scraper(source: str = "mock") -> BaseJobScraper:
    """工厂函数：获取对应平台的抓取器"""
    scrapers = {
        "mock": MockJobScraper,
        "boss": BossZhipinScraper,
        "zhipin": BossZhipinScraper,
        "zhaopin": ZhaopinScraper,
        "zhilian": ZhaopinScraper,
        "url": URLJobScraper,
    }
    scraper_cls = scrapers.get(source, MockJobScraper)
    return scraper_cls()
