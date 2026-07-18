// A small pill showing a skill that fired on an assistant response.
export default function SkillBadge({ name }) {
  return (
    <span className="inline-flex items-center rounded-full border border-itac-blue/20 bg-itac-blue-light px-2 py-0.5 text-xs font-medium text-itac-blue">
      {name}
    </span>
  );
}
